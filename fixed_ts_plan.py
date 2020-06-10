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
    params = SafeConfigParser()
    params.read('rl.ini')
   
    #Get traffic signal type
    signal_type = params.get('DEFAULT', 'signal')
    
    # Output csv file to save traffic metrics
    out_csv = 'outputs/{}-{}'.format(signal_type,'fixed_ts_plan')
    
    #Get the signal phases for the traffic network
    if signal_type == 'one_way':
        signal_phase = [traci.trafficlight.Phase(42000,-1,1, "GGrr"), 
                        traci.trafficlight.Phase(2000,-1,1, "yyrr"),
                        traci.trafficlight.Phase(42000,-1,1, "rrGG"),         
                        traci.trafficlight.Phase(2000,-1,1, "rryy")]
        
    elif signal_type == 'two_way':
        signal_phase = [traci.trafficlight.Phase(32000,-1,1, "GGrrrrGGrrrr"),
                        traci.trafficlight.Phase(2000,-1,1, "yyrrrryyrrrr"),
                        traci.trafficlight.Phase(32000,-1,1, "rrGrrrrrGrrr"),
                        traci.trafficlight.Phase(2000,-1,1, "rryrrrrryrrr"),
                        traci.trafficlight.Phase(32000,-1,1, "rrrGGrrrrGGr"),
                        traci.trafficlight.Phase(2000,-1,1, "rrryyrrrryyr"),
                        traci.trafficlight.Phase(32000,-1,1, "rrrrrGrrrrrG"),
                        traci.trafficlight.Phase(2000,-1,1, "rrrrryrrrrry")]

    #Initialize SUMO traffic simulation environment and get initial states
    env = SumoEnvironment(params,
                          out_csv_name=out_csv,
                          phases=signal_phase)
    initial_states = env.sumo_init()

    step = 0 
    current_phase = 0
    
    #3300 simulation steps equivalent to ~100000 sumo steps
    simulation_step = 3300
    
    #Start simulation
    while step < simulation_step:
        #Simulate SUMO at current phase for max_green time and get next green phase
        next_phase = env.fixed_sim(current_phase)
        current_phase = next_phase
        step +=1
        
    # Save and plot the traffic metrics:(step count, stopped vehicles, total wait time)
    save_csv(env.metrics, out_csv)
    plot(out_csv)
    
    env.close()