import gymnasium as gym, ale_py
gym.register_envs(ale_py)
env = gym.make("BreakoutNoFrameskip-v4")   # dueling probe: add full_action_space=True

