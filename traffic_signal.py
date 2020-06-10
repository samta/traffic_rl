import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci


class TrafficSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API
    """

    def __init__(self, env, ts_id, delta_time, min_green, max_green, phases):
        self.vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        self.id = ts_id
        self.env = env
        self.time_on_phase = 0.0
        self.delta_time = delta_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.num_green_phases = len(phases) // 2
        self.lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))  
        self.edges = self._compute_edges()
        self.edges_capacity = self._compute_edges_capacity()
        self.vehicles = {}

        logic = traci.trafficlight.Logic("new-program", 0, 0, 0, phases)
        #logic = traci.trafficlight.Logic("new-program", 0, 0, phases=phases)

        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.id, logic)

    def phase(self):
        """
        Returns current phase of the traffic light
        """
        return traci.trafficlight.getPhase(self.id)

    def set_next_phase(self, new_phase):
        """
        If current green phase time < min green time, sets new green phase to the same
        else Sets yellow phase before switching to next green phase
        """
        new_phase *= 2
        if self.phase == new_phase or self.time_on_phase < self.min_green:
            self.time_on_phase += self.delta_time
            self.green_phase = self.phase
        else:
            self.time_on_phase = self.delta_time
            self.green_phase = new_phase
            traci.trafficlight.setPhase(self.id, self.phase + 1)  # turns yellow

    def update_phase(self):
        """
        Update phase of the traffic light to the set green phase
        """
        traci.trafficlight.setPhase(self.id, self.green_phase)

    """
    Functions to collect data about state of the traffic environment 
    """
    def _compute_edges(self):
        return {p : self.lanes[p*2:p*2+2] for p in range(self.num_green_phases)}  

    def _compute_edges_capacity(self):
        return {p : sum([traci.lane.getLength(lane) for lane in self.edges[p]]) / self.vehicle_size_min_gap for p in range(self.num_green_phases)}

    def get_density(self):
        return [sum([traci.lane.getLastStepVehicleNumber(lane) for lane in self.edges[p]]) / self.edges_capacity[p] for p in range(self.num_green_phases)]

    def get_stopped_density(self):
        return [sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.edges[p]]) / self.edges_capacity[p] for p in range(self.num_green_phases)]

    def get_stopped_vehicles_num(self):
        return [sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.edges[p]]) for p in range(self.num_green_phases)]

    def get_waiting_time(self):
        wait_time_per_road = []
        for p in range(self.num_green_phases):
            veh_list = sum([list(traci.lane.getLastStepVehicleIDs(lane)) for lane in self.edges[p]], [])
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = traci.vehicle.getLaneID(veh)[:-2]
                acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.vehicles:
                    self.vehicles[veh] = {veh_lane: acc}
                else:
                    self.vehicles[veh][veh_lane] = acc - sum([self.vehicles[veh][lane] for lane in self.vehicles[veh].keys() if lane != veh_lane])
                wait_time += self.vehicles[veh][veh_lane]
            wait_time_per_road.append(wait_time)
        return wait_time_per_road

    def get_lanes_density(self):
        return [traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / self.vehicle_size_min_gap) for lane in self.lanes]
    
    def get_lanes_queue(self):
        return [traci.lane.getLastStepHaltingNumber(lane) / (traci.lane.getLength(lane) / self.vehicle_size_min_gap) for lane in self.lanes]
