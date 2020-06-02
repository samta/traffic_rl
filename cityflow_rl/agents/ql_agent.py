from agents.agent import Agent
from functools import reduce
import numpy as np
import logging

from exploration.epsilon_greedy import EpsilonGreedy


class QLAgent(Agent):

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        super(QLAgent, self).__init__(state_space, action_space)
        self.state = starting_state
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        print ('self.state:', self.state)
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.exploration = exploration_strategy
        self.acc_reward = 0

    def new_episode(self):
        pass

    def observe(self, observation):
        ''' To override '''
        pass

    def act(self, state):
        #logging.info('self.q_table:%s' % self.q_table)
        logging.info('new state %s' % self.state)
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        logging.info('self.action:%s' % self.action)
        return self.action

    def learn(self, new_state, reward, done=False):
        if new_state not in self.q_table:
            logging.info('state:{} not in q_table'.format(new_state))
            self.q_table[new_state] = [0 for _ in range(self.action_space.n)]
        else:
            logging.info('state:{} is in q_table'.format(new_state))


        s = self.state
        s1 = new_state
        a = self.action

        logging.info('q_table:%s' % self.q_table)
        logging.info('actiom:%s' % a)
        logging.info('q_table[s][a]:%s' % self.q_table[s][a])
        logging.info('new state s1 {}, old state s {}'.format(s1, s))
        logging.info('{} = {} + {} * ({} + {}* max({})) - {}'.format(self.q_table[s][a], self.q_table[s][a], self.alpha,
                                                                     reward, self.gamma, self.q_table[s1],
                                                                     self.q_table[s][a]))
        q = self.q_table[s][a] + self.alpha*(reward + self.gamma*max(self.q_table[s1]) - self.q_table[s][a])
        logging.info('update q at s:{} a:{} to {}'.format(s, a, q))
        self.q_table[s][a] = round(q, 2)
        self.state = s1
        self.acc_reward += reward
