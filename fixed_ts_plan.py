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
    #simulation_step = int(rl_params.get('DEFAULT', 'num_simulations'))

    # define output csv file
    experiment_time = str(datetime.now()).split('.')[0]
    out_csv = 'outputs/{}-{}'.format(experiment_time,'default_ts_plan')
    result = 'outputs/fixed_ts_plan'

    # init sumo environment
    rl_env = SumoEnvironment(rl_params,
                             out_csv_name=out_csv,
                             phases=[
                                 traci.trafficlight.Phase(32, "GGrrrrGGrrrr"),
                                 traci.trafficlight.Phase(2, "yyrrrryyrrrr"),
                                 traci.trafficlight.Phase(32, "rrGrrrrrGrrr"),
                                 traci.trafficlight.Phase(2, "rryrrrrryrrr"),
                                 traci.trafficlight.Phase(32, "rrrGGrrrrGGr"),
                                 traci.trafficlight.Phase(2, "rrryyrrrryyr"),
                                 traci.trafficlight.Phase(32, "rrrrrGrrrrrG"),
                                 traci.trafficlight.Phase(2, "rrrrryrrrrry")
                             ])

    # initialize the states
    initial_states = rl_env.reset()

    # initialize the agent
    rl_agent = rl_params.get('DEFAULT', 'rl_agent')

    step = 0 # initialize simulations step
    current_phase = 0
    simulation_step = 2000
    while step < simulation_step:
        #print ('simulation step')
        # get min green time
        # get the current phase
        next_phase = rl_env.sim(current_phase)
        #print ('next phase', next_phase)
        current_phase = next_phase
        # get time spent on current phase
        # if elapsed time > min green time
        # change the phase
        step += 1
        # save the metrics, step count and total wait time
    save_csv(rl_env.metrics, out_csv)
    # plot the metrics
    plot(out_csv, result)
    rl_env.close()