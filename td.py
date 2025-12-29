import numpy as np



class TemporalDifference():
    def __init__(self, env, n_iterations, gamma, lr):
        self.env = env
        self.n_iterations = n_iterations
        self.gamma = gamma
        self.lr = lr
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.Q = None
        self.history = []

    def td_learning(self):
        Q = np.zeros((self.n_states,self.n_actions))

        for i in range(self.n_iterations):
            current_state = self.env.reset()[0]
            done = False
            total_reward = 0
            while not done:
                policy = np.argmax(Q, axis=1)
                action = np.argmax(Q[current_state])
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                next_action = np.argmax(Q[next_state])
                target = reward + self.gamma * Q[next_state, next_action]
                delta = target - Q[current_state, action]
                Q[current_state, action] += self.lr * delta
                current_state = next_state
            
            # Track history every 100 episodes
            if i % 100 == 0:
                self.history.append({'episode': i, 'reward': total_reward})
        self.Q = Q
        policy = np.argmax(Q, axis=1)
        return policy