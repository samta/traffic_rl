import numpy as np
from gym import spaces
import random
import logging

class EpsilonGreedy:

    def __init__(self, initial_epsilon=1.0, min_epsilon=0.0, decay=0.99):
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def choose(self, q_table, state, action_space):
        logging.info('q_table:{}, state:{}, action_space:{}'.format(q_table, state, action_space))
        #if np.random.rand() < self.epsilon:
        if np.random.rand() < self.epsilon:
            logging.info('explore')
            action = int(action_space.sample())
            #action = random.choice(range(8))
            #print ('explore action to:', action)
        else:
            logging.info('exploit')
            action = np.argmax(q_table[state])
            print ('exploit action to:', action)

        self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)
        return action

    def choose1(self, q_table, state, action_space):
        #if np.random.rand() <= self.epsilon:
        if np.random.rand() <= 0.9:
            print ('explore!')
            logging.info('Explore!')
            return random.randrange(8)
        #act_values = self.model.predict(state)
        print ('exploit!')
        print ('act_values:', q_table[state])
        logging.info('Exploiting with q values %s' % q_table[state])
        return np.argmax(q_table[state])  # returns action

    def reset(self):
        self.epsilon = self.initial_epsilon
