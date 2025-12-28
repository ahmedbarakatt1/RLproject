import env1
import numpy as np


class policy_iteration():
    def __init__(self, env, gamma, n_iter):
        self.env = env
        self.gamma = gamma
        self.n_iter = n_iter
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.V = [0 for _ in range(self.num_states)]
        self.optimal_policy = list(np.random.randint(0, self.num_actions, size=self.num_states))

    def get_optimal_policy(self):
        for i in range(self.n_iter):
            for s in range(self.num_states):
                action = self.optimal_policy[s]
                state_data = self.env.unwrapped.P[s][action]  # prob, state, reward
                prob = np.array([state_data[i][0] for i in range(len(state_data))])
                r = np.array([state_data[i][2] for i in range(len(state_data))])

                self.V[s] = sum(prob * (r + self.gamma * np.array([self.V[state_data[i][1]] for i in range(len(state_data))])))
        #now we got the value function for current policy
                
                q_actions = [0 for _ in range(self.num_actions)]
                for action in range(self.num_actions):
                    state_data = self.env.unwrapped.P[s][action]  # prob, state, reward
                    prob = np.array([state_data[i][0] for i in range(len(state_data))])
                    r = np.array([state_data[i][2] for i in range(len(state_data))])
                    q = sum(prob * (r + self.gamma * np.array([self.V[state_data[i][1]] for i in range(len(state_data))])))
                    q_actions[action] = q
                self.optimal_policy[s] = np.argmax(q_actions) #optimal action to be taken to maximize the value
        return self.optimal_policy

current_state = env1.env.reset()
n_iter = 1000
tgamma = 0.9
pi = policy_iteration(env1.env, gamma=tgamma, n_iter=n_iter)
policy = pi.get_optimal_policy()
print("Optimal Policy: \n{}".format(np.array(policy)))

s=0
done=False
while not done:
    t=env1.env.step(int(policy[s]))
    s=int(t[0])
    done=t[2] or t[3]
    env1.env.render()
