# Traffic signal control using RL Agent
#### Setup
```
sudo apt-get install sumo sumo-tools sumo-doc
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```

#### Install
```
python3 setup.py install
```
#### Configuration
All configuration parameter to do an rl control are present in rl.ini file. By default it uses One-way Single intersection traffic network with Q-learning agent. This file can be modified to run the required configuration

#### Run
For RL based traffic light control,
```
python3 rl_control.py
```
For cyclic fixed time traffic light control,
```
python3 fixed_ts_plan.py
```

##### Algorithm
We have explored Q-learning, SARSA and Double Q-Learning algorithms for the agent to learn and optimize the total waiting time of vehicles in the signal.

Q-Learning Update:
```
Q(St, At) = Q(St, At) + alpha * [ (Rt+1) + max over a (Q(St+1, a))-Q(St, At)]
```

SARSA Update:
```
Q(St, At) = Q(St, At) + alpha * [ (Rt+1) + (Q(St+1, At+1))-Q(St, At)]
```

Double Q-Learning Update:
```
Update either of these with probability 0.5
Q1(St, At) = Q1(St, At) + alpha * [ (Rt+1) + Q2(St+1, max over a (Q1(St+1, a)))-Q1(St, At)]
or
Q2(St, At) = Q2(St, At) + alpha * [ (Rt+1) + Q1(St+1, max over a (Q2(St+1, a)))-Q2(St, At)]
```

##### State, Action and Rewards 
State  : [Current Phase, Elapsed time, Lane density, Queue length]

Reward : Previous waiting time - Current waiting time

Action : Choose the next green phase





