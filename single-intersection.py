import argparse
import os
import sys
import pandas as pd
from datetime import datetime
import colorama
import redis

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from env import SumoEnvironment
from agent import QLAgent
from epsilon_greedy import EpsilonGreedy


if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Q-Learning Single-Intersection""")
    prs.add_argument("-route", dest="route", type=str, required=True, help="Route definition xml file.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1.0, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-ns", dest="ns", type=int, default=42, required=False, help="Fixed green time for NS.\n")
    prs.add_argument("-we", dest="we", type=int, default=42, required=False, help="Fixed green time for WE.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=20000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-r", dest="reward", type=str, default='wait', required=False, help="Reward function: [-r queue] for average queue reward or [-r wait] for waiting time reward.\n")
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    args = prs.parse_args()
    ns = args.ns * 1000
    we = args.we * 1000
    experiment_time = str(datetime.now()).split('.')[0]
    out_csv = 'outputs/single-intersection/{}_alpha{}_gamma{}_eps{}_decay{}_reward{}'.format(experiment_time,
                                                                                                args.alpha, args.gamma,
                                                                                                args.epsilon,
                                                                                                args.decay, args.reward)

    env = SumoEnvironment(net_file='roadnets/single-intersection.net.xml',
                          route_file=args.route,
                          out_csv_name=out_csv,
                          use_gui=True,
                          num_seconds=args.seconds,
                          min_green=args.min_green,
                          max_green=args.max_green,
                          max_depart_delay=0,
                          phases=[
                            traci.trafficlight.Phase(ns, "GGrr"),   # north-south
                            traci.trafficlight.Phase(2000, "yyrr"),
                            traci.trafficlight.Phase(we, "rrGG"),   # west-east
                            traci.trafficlight.Phase(2000, "rryy")
                            ])

    print('reward:', args.reward)
    if args.reward == 'queue':
        env._compute_rewards = env._queue_average_reward
    else:
        env._compute_rewards = env._waiting_time_reward

    print('start run', args.runs)
    for run in range(1, args.runs+1):
        initial_states = env.reset()

        print('initial_states:', initial_states)
        print('env.observation_space:', env.observation_space)
        print('env.action_space:', env.action_space)

        ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts]),
                                 state_space=env.observation_space,
                                 action_space=env.action_space,
                                 alpha=args.alpha,
                                 gamma=args.gamma,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay)) for ts in env.ts_ids}

        done = {'__all__': False}
        infos = []
        if args.fixed:
            while not done['__all__']:
                _, _, done, _ = env.step({})
        else:
            while not done['__all__']:
                print ('ql_agents:', ql_agents)
                print ('ql_agents:', ql_agents.keys())
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
                print('action:', actions)
                s, r, done, _ = env.step(actions=actions)
                print ('s:{}, r:{}, done:{}'.format(s, r, done))
                print('s-1', env.encode(s['t']))
                print('s-2', s['t'])
                if args.v:
                    print('s=', env.radix_decode(ql_agents['t'].state), 'a=', actions['t'], 's\'=', env.radix_decode(env.encode(s['t'])), 'r=', r['t'])

                for agent_id in ql_agents.keys():
                    ql_agents[agent_id].learn(new_state=env.encode(s[agent_id]), reward=r[agent_id])
        print ('save env')
        env.save_csv()
        env.close()




