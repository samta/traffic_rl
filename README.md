# Trffic signal control using Q-Leaning
#### Setup
install sumo sumo-tools sumo-doc

echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc

source ~/.bashrc

#### Install
python3 setup.py install

#### Run
python3 rl_control.py

#### configuration
All configuration parameter to do an rl control is kept in rl.ini file. By default it uses single intersection one way traffic signal.

##### Algorithm:-
We have used Q-learning algorithm for the agent to learn and optimize the waiting time for vehicles in the signal.
```Q(St, At) = Q(St, At) + alpha*[Rt+1 + max over a (Q(St+1, a))-Q(St, At)]```

##### State:-
1. Current Phase - Current phase set on traffic signal
2. Elapsed time - Time spent on current signal
3. Lane density - last step vehicle on lane / (lane length / vehicle size (along with min gap between vehicle) 
4. Queue length - Last step halting/waiting vehicle on lane / (lane length / vehicle size (along with min gap between vehicle)

##### Reward:-
Total waiting time for all the vehicles on the traffic signal

##### Action:- 
which phase should be set green next

- single intersection, single road-single lane - phase - 0, 1 (action - 0/1)
- single intersection, two road - single lane - phase 0,1,2,3 (action - 0/1/2/3)
- single intersection, two road - double lane - phase 0,1,2,3,4,5,6,7 (action - 0/1/2/3/4/5/6/7)

Questions - what should be good state space, reward structure?




