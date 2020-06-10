import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np
plt.rc('lines', linewidth=2)
plt.rc('axes', prop_cycle=(cycler('color', ['#e41a1c','#377eb8','#4daf4a','#984ea3']))) # line colors


class encodeStates(object):
    def __init__(self, observation_space, num_green_phases, max_green, delta_time):
        self.num_green_phases = num_green_phases
        self.max_green = max_green
        self.delta_time = delta_time
        self.radix_factors = [s.n for s in observation_space.spaces]

    def encode(self, state):
        phase = state[:self.num_green_phases].index(1)
        elapsed = self.discretize_elapsed_time(state[self.num_green_phases])
        density_queue = [self.discretize_density(d) for d in state[self.num_green_phases + 1:]]
        encoded_state = self.radix_encode([phase, elapsed] + density_queue)
        return encoded_state

    def discretize_density(self, density):
        if density < 0.1:
            return 0
        elif density < 0.2:
            return 1
        elif density < 0.3:
            return 2
        elif density < 0.4:
            return 3
        elif density < 0.5:
            return 4
        elif density < 0.6:
            return 5
        elif density < 0.7:
            return 6
        elif density < 0.8:
            return 7
        elif density < 0.9:
            return 8
        else:
            return 9

    def discretize_elapsed_time(self, elapsed):
        elapsed *= self.max_green
        for i in range(self.max_green // self.delta_time):
            if elapsed <= self.delta_time + i * self.delta_time:
                return i
        return self.max_green // self.delta_time - 1

    def radix_encode(self, values):
        res = 0
        for i in range(len(self.radix_factors)):
            res = res * self.radix_factors[i] + values[i]
        return int(res)

def save_csv(metrics, out_csv_name):
    if out_csv_name is not None:
        df = pd.DataFrame(metrics)
        out_filename = out_csv_name
        df.to_csv(out_filename + '.csv', index=False)

def fig():
    fig = 1
    while True:
        yield fig
        fig += 1
fig_gen = fig()

def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def plot_figure(figsize=(12, 9), x_label='', y_label='', title=''):
    plt.figure(next(fig_gen), figsize=figsize)
    plt.rcParams.update({'font.size': 20})
    ax = plt.subplot()
    plt.grid(axis='y')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def plot(out_file):
    csv_file = out_file+'.csv'
    result_file = out_file+'.png'
    plot_figure(x_label='Time Step (s)', y_label='Total Waiting Time of Vehicles (s)')
    df = pd.read_csv(csv_file)
    main_df = df

    window = 5
    steps = main_df.groupby('step_time').total_stopped.mean().keys()
    mean = moving_average(main_df.groupby('step_time').mean()['total_wait_time'], window_size=window)
    std = moving_average(main_df.groupby('step_time').std()['total_wait_time'], window_size=window)

    plt.plot(steps, mean)
    plt.fill_between(steps, mean + std, mean - std, alpha=0.3)

    plt.savefig(result_file, bbox_inches="tight")