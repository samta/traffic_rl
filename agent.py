import numpy as np
from epsilon_greedy import EpsilonGreedy

class Agent():
    def __init__(self, starting_state, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        self.state = starting_state
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.q1_table = {self.state: [0 for _ in range(action_space.n)]}
        self.q2_table = {self.state: [0 for _ in range(action_space.n)]}
        self.exploration = exploration_strategy

    def act(self, step):
        '''
        Agent picking an action based on the exploration scheme
        '''
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        return self.action

    def learn_q(self, new_state, reward):
        '''
        Q-Learning Agent
        '''
        if new_state not in self.q_table:
            self.q_table[new_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        a = self.action
        s1 = new_state
        self.q_table[s][a] = self.q_table[s][a] + self.alpha*(reward + self.gamma*max(self.q_table[s1]) - self.q_table[s][a])
        self.state = s1

    def learn_sarsa(self, new_state, reward):
        '''
        SARSA Agent
        '''
        if new_state not in self.q_table:
            self.q_table[new_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        a = self.action
        self.q_table[s][a] = self.q_table[s][a] + self.alpha*(reward + self.gamma*(self.q_table[new_state][self.act(new_state)]) - self.q_table[s][a])
        self.state = new_state
        
    def learn_dql(self, new_state, reward):
        '''
        Double Q-Learning Agent
        '''
        if new_state not in self.q_table:
            self.q_table[new_state] = [0 for _ in range(self.action_space.n)]
            self.q1_table[new_state] = [0 for _ in range(self.action_space.n)]
            self.q2_table[new_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        a = self.action
        s1 = new_state
        
        #update q1(s,a) or q2(s,a) with probability of 0.5
        if np.random.rand() < 0.5:
            self.q1_table[s][a] += self.alpha*(reward + self.gamma*(self.q2_table[s1][np.argmax(self.q1_table[s1])]) - self.q1_table[s][a])
        else:
            self.q2_table[s][a] += self.alpha*(reward + self.gamma*(self.q1_table[s1][np.argmax(self.q2_table[s1])]) - self.q2_table[s][a])
        self.q_table[s][a] = self.q1_table[s][a] + self.q2_table[s][a]
        self.state = s1
