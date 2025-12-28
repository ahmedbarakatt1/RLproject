import env1
import numpy as np


class monte_carlo():
    def __init__(self, env, gamma, n_episodes):
        self.env = env
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.returns_count = np.zeros((self.num_states, self.num_actions))
        self.optimal_policy = [0 for _ in range(self.num_states)]