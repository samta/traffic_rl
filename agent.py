from epsilon_greedy import EpsilonGreedy


class Agent():
    def __init__(self, starting_state, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        self.state = starting_state
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.exploration = exploration_strategy

    def act(self, step):
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        return self.action

    def learn_q(self, new_state, reward):
        if new_state not in self.q_table:
            self.q_table[new_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        a = self.action
        s1 = new_state
        self.q_table[s][a] = self.q_table[s][a] + self.alpha*(reward + self.gamma*max(self.q_table[s1]) - self.q_table[s][a])
        self.state = s1
        #self.action = a1

    def learn_sarsa(self, new_state, reward):
        pass
