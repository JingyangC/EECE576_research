# EECE576 CleanRL DQN baseline with periodic evaluation.
#
# Configuration:
#   (PER, Dueling, Multi-step) = (0, 0, 0)
#
# The underlying DQN uses the adopted CleanRL hyperparameters.
# Training/evaluation budgets are reduced to fit Kaggle compute limits.
#
# Current experimental settings:
#   - 84x84 grayscale observations
#   - 4 stacked frames
#   - action repeat 4
#   - reward clipping during training
#   - replay capacity 250,000 transitions
#   - batch size 32
#   - gamma 0.99
#   - learning update every 4 agent steps
#   - learning starts after 80,000 agent steps
#   - target network update every 1,000 agent steps
#   - Adam learning rate 1e-4
#   - epsilon 1.0 -> 0.01 over first 10% of training
#   - total training: 5M agent steps ~= 20M ALE frames
#   - evaluation every 100K agent steps ~= 400K ALE frames
#   - each evaluation lasts 37.5K agent steps ~= 150K ALE frames
#   - evaluation epsilon = 0.05

import os
import random
import time
from dataclasses import dataclass

import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.buffers import ReplayBuffer
from collections import deque
from typing import NamedTuple

FRAME_SKIP = 4

@dataclass
class Args:
    """Experiment and DQN hyperparameters."""

    # General experiment settings
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    # W&B / TensorBoard
    track: bool = False
    wandb_project_name: str = "EECE576-DQN"
    wandb_entity: str | None = None
    capture_video: bool = False

    # Saving / session management
    save_model: bool = False
    checkpoint_frequency: int = 250_000
    max_hours: float = 0.0

    # Periodic evaluation
    eval_frequency: int = 100_000
    eval_steps: int = 37_500
    eval_epsilon: float = 0.05

    # Environment
    env_id: str = "BreakoutNoFrameskip-v4"
    num_envs: int = 1

    # 20M ALE frames / action-repeat 4 = 5M agent steps
    total_timesteps: int = 5_000_000

    # cleanRL common backbone
    learning_rate: float = 1e-4
    buffer_size: int = 250_000
    gamma: float = 0.99
    batch_size: int = 32

    # Hard target-network update:
    target_network_frequency: int = 1_000
    tau: float = 1.0

    # CleanRL epsilon schedule: decay over first 10% of total training
    # For 5M steps, epsilon reaches 0.01 after 500K agent steps
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.10

    # Original CleanRL setting: start learning after 80K agent steps
    learning_starts: int = 80_000

    # One gradient update every 4 agent decisions
    train_frequency: int = 4

    # Rainbow components used in the factorial ablation
    use_per: bool = False
    use_dueling: bool = False
    use_multistep: bool = False

    # For multi-step
    n_step: int = 3

    # Prioritized Experience Replay
    per_alpha: float = 0.5
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_eps: float = 1e-6

def make_env(env_id, seed, idx, capture_video, run_name):
    """Training environment using DQN/Rainbow Atari preprocessing."""

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        # Keep this inside the reward-clipping wrapper so episode statistics
        # reflect the underlying game score, while the learner receives
        # clipped rewards.
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Standard Atari preprocessing used by DQN/Rainbow.
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=FRAME_SKIP)
        env = EpisodicLifeEnv(env)

        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        env.action_space.seed(seed)

        return env

    return thunk

def make_eval_env(env_id, seed):
    """Evaluation environment for raw full-game scores.

    Evaluation keeps the same observation preprocessing and action repeat,
    but it does NOT clip rewards and it does NOT treat a lost life as the end
    of the whole game. This lets eval/mean_return represent raw game score.
    """
    env = gym.make(env_id)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=FRAME_SKIP)

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)

    # Outermost wrapper: one recorded evaluation episode = one full game.
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)

    return env

class QNetwork(nn.Module):
    """Standard DQN convolutional network; Dueling is OFF."""

    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)

class DuelingQNetwork(nn.Module):
    """Dueling DQN convolutional network based on CleanRL Rainbow Algorithm; Dueling is ON."""

    def __init__(self, env):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        features = self.features(x / 255.0)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        return value + advantage - advantage.mean(dim=1, keepdim=True)

class PrioritizedReplaySamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    indices: np.ndarray
    weights: torch.Tensor

class SumTree:
    """Binary sum tree for proportional prioritized replay."""

    def __init__(self, capacity):
        tree_capacity = 1

        while tree_capacity < capacity:
            tree_capacity *= 2

        self.capacity = tree_capacity
        self.tree = np.zeros(2 * self.capacity, dtype=np.float32)

    def update(self, index, priority):
        tree_index = index + self.capacity
        self.tree[tree_index] = priority
        tree_index //= 2

        while tree_index >= 1:
            self.tree[tree_index] = (self.tree[2 * tree_index] + self.tree[2 * tree_index + 1])
            tree_index //= 2

    def total(self):
        return self.tree[1]

    def get(self, value):
        tree_index = 1

        while tree_index < self.capacity:
            left = 2 * tree_index

            if value < self.tree[left]:
                tree_index = left
            else:
                value -= self.tree[left]
                tree_index = left + 1

        return tree_index - self.capacity

    def priority(self, index):
        return self.tree[index + self.capacity]

class PrioritizedReplayBuffer(ReplayBuffer):
    """Memory-efficient proportional Prioritized Experience Replay."""

    def __init__(self, *args, alpha=0.5, beta=0.4, eps=1e-6, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        # Store the maximum un-exponentiated priority.
        self.max_priority = 1.0
        self.sum_tree = SumTree(self.buffer_size)

    def add(self, obs, next_obs, action, reward, done, infos):
        # Remember which replay slot is about to be written.
        insert_index = self.pos
        super().add(obs, next_obs, action, reward, done, infos,)

        # New experiences receive maximum priority so that
        # they get sampled at least once.
        priority = self.max_priority ** self.alpha

        self.sum_tree.update(insert_index, priority)

        # With optimize_memory_usage=True, self.pos is not
        # currently a valid transition after the buffer is full.
        if self.optimize_memory_usage and self.full:
            self.sum_tree.update(self.pos, 0.0)

    def sample(self, batch_size):
        total_priority = self.sum_tree.total()

        if total_priority <= 0:
            raise RuntimeError("Cannot sample from an empty prioritized replay buffer.")

        indices = []

        # Divide the priority range into equal sections.
        # Sample once from each section.
        segment = total_priority / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            index = self.sum_tree.get(value)
            indices.append(index)

        indices = np.asarray(indices, dtype=np.int64)

        # Let the existing CleanRL ReplayBuffer retrieve
        # observations/actions/rewards/etc.
        data = self._get_samples(indices)

        priorities = np.asarray([self.sum_tree.priority(index) for index in indices], dtype=np.float32)
        probabilities = priorities / total_priority

        # Importance-sampling correction.
        weights = (self.size() * probabilities) ** (-self.beta)

        # Normalize so the largest batch weight is 1.
        weights /= weights.max()

        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        return PrioritizedReplaySamples(
            observations=data.observations,
            actions=data.actions,
            next_observations=data.next_observations,
            dones=data.dones,
            rewards=data.rewards,
            indices=indices,
            weights=weights,
        )

    def update_priorities(self, indices,  td_errors):
        priorities = (np.abs(td_errors) + self.eps)

        self.max_priority = max(self.max_priority, float(priorities.max()))

        for index, priority in zip(indices, priorities):
            scaled_priority = (float(priority) ** self.alpha)
            self.sum_tree.update(int(index), scaled_priority)

def get_n_step_transition(n_step_buffer, gamma, n_step):
    """Build one n-step transition from the front of the buffer."""

    first_obs, first_action, _, _, _, _ = n_step_buffer[0]

    total_reward = np.zeros_like(n_step_buffer[0][2])
    next_obs = n_step_buffer[0][3]
    done = n_step_buffer[0][4]

    for i, transition in enumerate(list(n_step_buffer)[:n_step]):
        _, _, reward, transition_next_obs, terminated, truncated = transition

        total_reward += (gamma ** i) * reward
        next_obs = transition_next_obs
        done = terminated

        # Do not let an n-step transition cross an episode boundary.
        if np.any(terminated) or np.any(truncated):
            break

    return (first_obs, first_action, total_reward, next_obs, done)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Linearly anneal epsilon from start_e to end_e."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def evaluate_for_steps(model, env_id, seed, eval_steps, device, epsilon):
    """Evaluate the frozen policy for a fixed number of agent steps.

    No learning or replay-buffer updates occur here. Returns the raw game
    scores of complete games that finish within the evaluation window.
    """
    env = make_eval_env(env_id, seed)
    obs, _ = env.reset(seed=seed)

    # Dedicated RNG so evaluation does not perturb the training Python RNG.
    eval_rng = random.Random(seed)

    episodic_returns = []
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for _ in range(eval_steps):
                if eval_rng.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    obs_tensor = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = model(obs_tensor)
                    action = int(torch.argmax(q_values, dim=1).item())

                next_obs, _, terminated, truncated, info = env.step(action)
                obs = next_obs

                if terminated or truncated:
                    if "episode" in info:
                        episodic_return = float(np.asarray(info["episode"]["r"]).item())
                        episodic_returns.append(episodic_return)

                    obs, _ = env.reset()
    finally:
        env.close()

        if was_training:
            model.train()

    return episodic_returns


if __name__ == "__main__":
    args = tyro.cli(Args)

    assert args.num_envs == 1, ("This implementation currently supports exactly one training env.")

    config_name = (
        f"P{int(args.use_per)}_"
        f"D{int(args.use_dueling)}_"
        f"M{int(args.use_multistep)}"
    )

    print(f"Configuration: {config_name}")

    run_name = (
        f"{args.env_id}__"
        f"{config_name}__"
        f"seed{args.seed}__"
        f"{int(time.time())}"
    )

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and args.cuda
        else "mps"
        if torch.backends.mps.is_available() and args.cuda
        else "cpu"
    )

    print(f"device={device}")

    # Training environment
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name,) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only discrete action spaces are supported."

    # ------------------------------------------------------------------
    # (PER, Dueling, Multi-step) = (0, 0, 0)
    # ------------------------------------------------------------------

    # Dueling = ON/OFF
    NetworkClass = (
        DuelingQNetwork
        if args.use_dueling
        else QNetwork
    )

    q_network = NetworkClass(envs).to(device)
    target_network = NetworkClass(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Optimizer settings.
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    # PER = On/OFF -> PER/uniform replay.
    if args.use_per:
        rb = PrioritizedReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
            alpha=args.per_alpha,
            beta=args.per_beta_start,
            eps=args.per_eps,
        )
    else:
        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )

    start_time = time.time()

    model_dir = f"runs/{run_name}"
    model_path = f"{model_dir}/{args.exp_name}.cleanrl_model"
    best_model_path = f"{model_dir}/{args.exp_name}.best.cleanrl_model"
    best_eval_return = -float("inf")

    obs, _ = envs.reset(seed=args.seed)

    n_step_buffer = deque()

    for global_step in range(args.total_timesteps):
        # global_step is zero-based; agent_step is easier to interpret.
        agent_step = global_step + 1

        if args.use_per:
            progress = global_step / args.total_timesteps
            rb.beta = (args.per_beta_start + progress * (args.per_beta_end - args.per_beta_start))
            rb.beta = min(rb.beta, args.per_beta_end,)

        # Approximate raw ALE frames, because each selected action is
        # repeated for FRAME_SKIP=4 emulator frames.
        ale_frames = agent_step * FRAME_SKIP

        # --------------------------------------------------------------
        # Action selection
        # --------------------------------------------------------------
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)

        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                q_values = q_network(torch.as_tensor( obs, dtype=torch.float32, device=device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # One agent decision advances ~4 ALE frames.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Training return is useful for monitoring, but the main
        # experimental curve is eval/mean_return.
        if "final_info" in infos and "episode" in infos["final_info"]:
            final_info = infos["final_info"]

            for idx in np.flatnonzero(final_info["_episode"]):
                episodic_return = final_info["episode"]["r"][idx]
                episodic_length = final_info["episode"]["l"][idx]

                print(
                    f"agent_step={agent_step}, "
                    f"ale_frames={ale_frames}, "
                    f"episodic_return={episodic_return}"
                )

                writer.add_scalar("charts/episodic_return", episodic_return, ale_frames)
                writer.add_scalar("charts/episodic_length", episodic_length, ale_frames)

        # --------------------------------------------------------------
        # Replay storage
        # Uniform or prioritized depending on args.use_per
        # --------------------------------------------------------------
        real_next_obs = next_obs.copy()

        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_obs"][idx]

        if args.use_multistep:
            n_step_buffer.append((obs.copy(), actions.copy(), rewards.copy(), real_next_obs.copy(), terminations.copy(), truncations.copy()))

            # Once we have n transitions, construct one n-step sample.
            if len(n_step_buffer) >= args.n_step:
                n_obs, n_action, n_reward, n_next_obs, n_done = get_n_step_transition(n_step_buffer, args.gamma, args.n_step)
                rb.add(n_obs, n_next_obs, n_action, n_reward, n_done, infos)
                n_step_buffer.popleft()

            # At an episode boundary, flush the remaining transitions.
            if np.any(terminations):
                while n_step_buffer:
                    n_obs, n_action, n_reward, n_next_obs, n_done = get_n_step_transition(n_step_buffer, args.gamma, args.n_step)
                    rb.add(n_obs, n_next_obs, n_action, n_reward, n_done, infos)
                    n_step_buffer.popleft()

            elif np.any(truncations):
                n_step_buffer.clear()

        else:
            # Original 1-step DQN behavior.
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        # --------------------------------------------------------------
        # 1-step or n-step DQN update
        # Multi-step controlled by args.use_multistep
        # --------------------------------------------------------------
        if agent_step > args.learning_starts:
            if agent_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)

                    # n-step TD target:
                    if args.use_multistep:
                        bootstrap_gamma = args.gamma ** args.n_step
                    else:
                        bootstrap_gamma = args.gamma

                    td_target = (data.rewards.flatten() + bootstrap_gamma * target_max * (1 - data.dones.flatten()))

                old_val = q_network(data.observations).gather(1, data.actions,).squeeze()
                td_error = td_target - old_val

                if args.use_per:
                    loss = (data.weights * td_error.pow(2)).mean()
                    rb.update_priorities(data.indices, td_error.detach().cpu().numpy())

                else:
                    loss = F.mse_loss(td_target, old_val)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if agent_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss.item(), ale_frames)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), ale_frames)
                    sps = int(agent_step / (time.time() - start_time))
                    writer.add_scalar("charts/SPS", sps, ale_frames)
                    writer.add_scalar("charts/epsilon", epsilon, ale_frames)

                    if args.use_per:
                        writer.add_scalar("charts/per_beta", rb.beta, ale_frames)

                    if agent_step % 10_000 == 0:
                        print("SPS:", sps)

            # Hard target-network update every 1K agent steps
            # after learning has started.
            steps_since_learning_start = ( agent_step - args.learning_starts)

            if (steps_since_learning_start > 0 and steps_since_learning_start % args.target_network_frequency == 0):
                for target_param, online_param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(args.tau * online_param.data + (1.0 - args.tau) * target_param.data)

        # --------------------------------------------------------------
        # Periodic evaluation
        # --------------------------------------------------------------
        if (args.eval_frequency > 0 and agent_step % args.eval_frequency == 0):
            eval_returns = evaluate_for_steps(
                model=q_network,
                env_id=args.env_id,
                seed=args.seed + 10_000,
                eval_steps=args.eval_steps,
                device=device,
                epsilon=args.eval_epsilon,
            )

            if eval_returns:
                mean_eval_return = float(np.mean(eval_returns))
                std_eval_return = float(np.std(eval_returns))

                print(
                    f"[EVAL] agent_step={agent_step}, "
                    f"ale_frames={ale_frames}, "
                    f"episodes={len(eval_returns)}, "
                    f"mean_return={mean_eval_return:.3f}, "
                    f"std_return={std_eval_return:.3f}"
                )

                writer.add_scalar("eval/mean_return", mean_eval_return, ale_frames)
                writer.add_scalar("eval/std_return", std_eval_return, ale_frames)
                writer.add_scalar("eval/episodes_completed", len(eval_returns), ale_frames)
                writer.flush()

                if (args.save_model and mean_eval_return > best_eval_return):
                    best_eval_return = mean_eval_return
                    torch.save( q_network.state_dict(), best_model_path + ".tmp")
                    os.replace(best_model_path + ".tmp", best_model_path)

                    print(
                        "[EVAL] new best evaluated model saved to "
                        f"{best_model_path}"
                    )
            else:
                print(
                    f"[EVAL] agent_step={agent_step}: "
                    "no complete games finished in the "
                    "evaluation window"
                )

        # --------------------------------------------------------------
        # Periodic checkpoint
        # --------------------------------------------------------------
        if args.save_model and agent_step % args.checkpoint_frequency == 0:
            torch.save(q_network.state_dict(), model_path + ".tmp")
            os.replace(model_path + ".tmp", model_path)

        # --------------------------------------------------------------
        # Optional session time limit
        # --------------------------------------------------------------
        if args.max_hours > 0 and time.time() - start_time > args.max_hours * 3600:
            print(
                f"reached max_hours={args.max_hours} "
                f"at agent_step={agent_step}, "
                f"ale_frames={ale_frames}; stopping"
            )
            break

    if args.save_model:
        torch.save(q_network.state_dict(), model_path)
        print(f"latest model saved to {model_path}")

        if os.path.exists(best_model_path):
            print(
                f"best evaluated model saved to "
                f"{best_model_path}"
            )

    envs.close()
    writer.close()