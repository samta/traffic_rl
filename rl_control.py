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
from util import save_csv, plot

if __name__ == '__main__':

    # load default config
    rl_params = SafeConfigParser()
    rl_params.read('rl.ini')
    simulation_step = int(rl_params.get('DEFAULT', 'num_simulations'))

    # define output csv file
    experiment_time = str(datetime.now()).split('.')[0]
    out_csv = 'outputs/{}_{}_{}Agent'.format(experiment_time,
                                                                         rl_params.get('DEFAULT', 'signal'),
                                                                         rl_params.get('DEFAULT', 'rl_agent')
                                                                         )

    # init sumo environment
    signal_type = rl_params.get('DEFAULT', 'signal')

    #Get the signal phases for the traffic network
    if signal_type == 'one_way':
        signal_phase = [traci.trafficlight.Phase(42,"GGrr"),
                        traci.trafficlight.Phase(2,"yyrr"),
                        traci.trafficlight.Phase(42,"rrGG"),
                        traci.trafficlight.Phase(2,"rryy")]
        
    elif signal_type == 'two_way':
        signal_phase = [traci.trafficlight.Phase(32,"GGrrrrGGrrrr"),
                        traci.trafficlight.Phase(2,"yyrrrryyrrrr"),
                        traci.trafficlight.Phase(32,"rrGrrrrrGrrr"),
                        traci.trafficlight.Phase(2,"rryrrrrryrrr"),
                        traci.trafficlight.Phase(32,"rrrGGrrrrGGr"),
                        traci.trafficlight.Phase(2,"rrryyrrrryyr"),
                        traci.trafficlight.Phase(32,"rrrrrGrrrrrG"),
                        traci.trafficlight.Phase(2,"rrrrryrrrrry")]

    #Initialize SUMO traffic simulation environment and get initial states
    rl_env = SumoEnvironment(rl_params,
                             out_csv_name=out_csv,
                             phases=signal_phase)
    initial_states = rl_env.sumo_init()

    #Initialize the RL agent
    rl_agent = rl_params.get('DEFAULT', 'rl_agent')
    agent = Agent(starting_state=rl_env.encode_states(initial_states),
                             action_space=rl_env.action_space,
                             alpha=float(rl_params.get('DEFAULT', 'alpha')),
                             gamma=float(rl_params.get('DEFAULT', 'gamma')),
                             exploration_strategy=EpsilonGreedy(initial_epsilon=float(rl_params.get('DEFAULT', 'epsilon')),
                                                                min_epsilon=float(rl_params.get('DEFAULT', 'minimum_epsilon')),
                                                                decay=float(rl_params.get('DEFAULT', 'decay')))
                    )
    
    #Start simulation
    step = 0 
    while step < simulation_step:
        # Take a step
        action = agent.act(step)
        step += 1
        # Compute next_state and reward
        next_state, reward = rl_env.step(actions=action)
        if rl_agent == 'ql':
            # Apply Q-Learning
            agent.learn_q(new_state=rl_env.encode_states(next_state), reward=reward)
        elif rl_agent == 'sarsa':
            # Apply sarsa learning
            agent.learn_sarsa(new_state=rl_env.encode_states(next_state), reward=reward)
        elif rl_agent == 'dql':
            # Apply dql learning
            agent.learn_dql(new_state=rl_env.encode_states(next_state), reward=reward)

    # Save and plot the traffic metrics:(step count, stopped vehicles, total wait time)
    save_csv(rl_env.metrics, out_csv)
    plot(out_csv)
    
    rl_env.close()

