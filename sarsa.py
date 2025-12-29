import numpy as np


class SARSA():
    def __init__(self, n_iterations, gamma, lr, epsilon, env):
        self.env = env
        self.n_iterations = n_iterations
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.Q = None
        self.history = []

    def sarsa(self):
        Q = np.zeros((self.n_states, self.n_actions))
        current_epsilon = self.epsilon
        for i in range(self.n_iterations):
            current_state = self.env.reset()[0]
            # Select initial action
            if np.random.rand() < current_epsilon:
                action = np.random.randint(0, self.n_actions)
            else:
                action = np.argmax(Q[current_state])
            done = False
            total_reward = 0
            while not done:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                # Select next action epsilon-greedy
                if np.random.rand() < current_epsilon:
                    next_action = np.random.randint(0, self.n_actions)
                else:
                    next_action = np.argmax(Q[next_state])
                #terminal state handling
                if done:
                    target = reward  # No future reward at terminal state
                else:
                    target = reward + self.gamma * Q[next_state, next_action]
                Q[current_state, action] += self.lr * (target - Q[current_state, action])
                current_state = next_state
                action = next_action
            current_epsilon = max(0.01, current_epsilon * 0.9995)
            # Track history every 100 episodes
            if i % 100 == 0:
                self.history.append({'episode': i, 'reward': total_reward})
        self.Q = Q
        policy = np.argmax(Q, axis=1)
        return policy
            