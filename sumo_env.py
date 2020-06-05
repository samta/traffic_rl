import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import matplotlib.pyplot as plt
import numpy as np

from traffic_signal import TrafficSignal
from util import encodeStates


class SumoEnvironment(MultiAgentEnv):
    """
    SUMO Environment for Traffic Signal Control
    """
    def __init__(self, params, phases, out_csv_name=None):

        self.net = params.get('DEFAULT', 'network_path')
        self.route = params.get('DEFAULT', 'route_path')
        self.gui = params.get('DEFAULT', 'use_gui')
        if self.gui.upper() == 'YES':
            self.sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self.sumo_binary = sumolib.checkBinary('sumo')

        traci.start([sumolib.checkBinary('sumo'), '-n', self.net])  # start only to retrieve information

        self.ts_id = traci.trafficlight.getIDList()[0]
        self.lanes_per_ts = len(set(traci.trafficlight.getControlledLanes(self.ts_id)))
        self.traffic_signals = None
        self.phases = phases
        self.num_green_phases = len(phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
        self.last_measure = 0.0  # used to reward function remember last measure
        self.time_to_load_vehicles = 120  # number of simulation seconds ran in reset() before learning starts
        self.delta_time = 5  # seconds on sumo at each step
        self.max_depart_delay = 0  # Max wait time to insert a vehicle
        self.min_green = int(params.get('DEFAULT', 'minimum_green_time'))
        self.max_green = int(params.get('DEFAULT', 'maximum_green_time'))
        self.yellow_time = int(params.get('DEFAULT', 'yellow_time'))
        # define the observation space
        self.observation_space = spaces.Box(low=np.zeros(self.num_green_phases + 1 + 2*self.lanes_per_ts),
                                            high=np.ones(self.num_green_phases + 1 + 2*self.lanes_per_ts))
        self.discrete_observation_space = spaces.Tuple((
            spaces.Discrete(self.num_green_phases),
            spaces.Discrete(self.max_green//self.delta_time),
            *(spaces.Discrete(10) for _ in range(2*self.lanes_per_ts))
        ))
        self.action_space = spaces.Discrete(self.num_green_phases)

        self.spec = ''

        self.metrics = []
        self.out_csv_name = out_csv_name
        traci.close()

    def sumo_step(self):
        traci.simulationStep()


    def encode_states(self, state):
        encode_obj = encodeStates(self.discrete_observation_space,
                     self.num_green_phases,
                     self.max_green,
                     self.delta_time)

        return encode_obj.encode(state)
        
    def reset(self):
        self.metrics = []

        sumo_cmd = [self.sumo_binary,
                     '-n', self.net,
                     '-r', self.route,
                     '--max-depart-delay', str(self.max_depart_delay), 
                     '--waiting-time-memory', '10000',
                     '--time-to-teleport', str(-1),
                     '--random']
        traci.start(sumo_cmd)

        self.traffic_signals = TrafficSignal('',self.ts_id, self.delta_time, self.min_green, self.max_green, self.phases)
        self.last_measure = 0.0

        # Load vehicles
        self.sumo_step()

        return self.compute_step()

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        #return traci.simulation.getCurrentTime()/1000  # milliseconds to seconds
        return traci.simulation.getTime()  # milliseconds to seconds

    def step(self, actions):
        # act, apply action
        self.apply_actions(actions)
        for _ in range(self.yellow_time):
            ##print('step till yellow light')
            # apply sumo step till yellow light
            self.sumo_step()
        self.traffic_signals.update_phase()
        for _ in range(self.delta_time - self.yellow_time):
            self.sumo_step()

        # observe new state and reward
        step_info = self.compute_step()
        reward = self.compute_rewards()
        info = self.compute_step_info()
        self.metrics.append(info)

        return step_info, reward

    def apply_actions(self, actions):
        """
        Set the next green phase for the traffic signals
        """
        self.traffic_signals.set_next_phase(actions)

    def compute_step(self):
        """
        Return the current observation for each traffic signal
        """
        phase_id = [1 if self.traffic_signals.phase//2 == i else 0 for i in range(self.num_green_phases)]
        elapsed = self.traffic_signals.time_on_phase / self.max_green
        density = self.traffic_signals.get_lanes_density()
        queue = self.traffic_signals.get_lanes_queue()
        observations = phase_id + [elapsed] + density + queue
        return observations

    def compute_rewards(self):
        ts_wait = sum(self.traffic_signals.get_waiting_time())
        rewards = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return rewards

    def compute_step_info(self):
        return {
            'step_time': self.sim_step,
            'total_stopped': sum(self.traffic_signals.get_stopped_vehicles_num()),
            'total_wait_time': self.last_measure
        }

    def close(self):
        traci.close()

    def sim(self, phase):
        time_on_phase = 0.0
        max_green = 50
        #print ('curr phase', phase)
        traci.trafficlight.setPhase('t', phase)  # turns yellow
        #self.traffic_signals.set_next_phase(phase)
        while time_on_phase <= max_green:
            #print ('self.traffic_signals.time_on_phase', time_on_phase)
            #print ('apply step')
            self.sumo_step()
            time_on_phase += 1

        self.last_measure = sum(self.traffic_signals.get_waiting_time())
        m = self.compute_step_info()
        #print ('step info', m)
        self.metrics.append(m)

        if phase == 7: return 0
        #print ('>>traci.trafficlight.setPhase', traci.trafficlight.getPhase('t'))
        return (phase) + 1






