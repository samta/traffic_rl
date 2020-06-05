import os
import sys
from datetime import datetime

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from sumo_env import SumoEnvironment
from agent import Agent
from epsilon_greedy import EpsilonGreedy
from configparser import SafeConfigParser
from util import save_csv, plot, save_csv_1

if __name__ == '__main__':

    # load default config
    rl_params = SafeConfigParser()
    rl_params.read('rl.ini')
    simulation_step = int(rl_params.get('DEFAULT', 'num_simulations'))

    # define output csv file
    experiment_time = str(datetime.now()).split('.')[0]
    out_csv = 'outputs/{}'.format(experiment_time)
    result = 'outputs/result'
    # init sumo environment

    signal_type = rl_params.get('DEFAULT', 'signal')

    if signal_type == 'one_way':
        signal_phase = [traci.trafficlight.Phase(42, "GGrr"),  # north-south
                  traci.trafficlight.Phase(2, "yyrr"),
                  traci.trafficlight.Phase(42, "rrGG"),  # west-east
                  traci.trafficlight.Phase(2, "rryy")
                 ]
    elif signal_type == "two_way":
        signal_phase = [traci.trafficlight.Phase(32, "GGrrrrGGrrrr"),
                    traci.trafficlight.Phase(2, "yyrrrryyrrrr"),
                    traci.trafficlight.Phase(32, "rrGrrrrrGrrr"),
                    traci.trafficlight.Phase(2, "rryrrrrryrrr"),
                    traci.trafficlight.Phase(32, "rrrGGrrrrGGr"),
                    traci.trafficlight.Phase(2, "rrryyrrrryyr"),
                    traci.trafficlight.Phase(32, "rrrrrGrrrrrG"),
                    traci.trafficlight.Phase(2, "rrrrryrrrrry")
                 ]

    rl_env = SumoEnvironment(rl_params,
                             out_csv_name=out_csv,
                             phases=signal_phase)

    # initialize the states
    initial_states = rl_env.reset()

    # initialize the agent
    rl_agent = rl_params.get('DEFAULT', 'rl_agent')

    agent = Agent(starting_state=rl_env.encode_states(initial_states),
                             action_space=rl_env.action_space,
                             alpha=float(rl_params.get('DEFAULT', 'alpha')),
                             gamma=float(rl_params.get('DEFAULT', 'gamma')),
                             exploration_strategy=EpsilonGreedy(initial_epsilon=float(rl_params.get('DEFAULT', 'epsilon')),
                                                                min_epsilon=float(rl_params.get('DEFAULT', 'minimum_epsilon')),
                                                                decay=float(rl_params.get('DEFAULT', 'decay')))
                    )
    step = 0 # initialize simulations step
    while step < simulation_step:
        # take a step
        action = agent.act(step)
        step += 1
        # compute next_state and reward
        next_state, reward = rl_env.step(actions=action)
        if rl_agent == 'ql':
            # Apply Q-Learning
            agent.learn_q(new_state=rl_env.encode_states(next_state), reward=reward)
            # Apply sarsa learnign
        elif rl_agent == 'sarsa':
            agent.learn_sarsa(new_state=rl_env.encode_states(next_state), reward=reward)

    # save the metrics, step count and total wait time
    save_csv(rl_env.metrics, out_csv)
    # plot the metrics
    plot(out_csv, result)
    #save_csv_1(rl_env.metrics)
    rl_env.close()

