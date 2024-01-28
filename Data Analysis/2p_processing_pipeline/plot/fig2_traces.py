#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

from modules import RetrieveResults


def plot_fig2(
        ops,
        ch = 'fluo_ch2',
        trial = 3,
        num = 20,
        start = 0,
        length = 1000,
        ):
    # read channel signal.
    [neural_trial, _] = RetrieveResults.run(ops)
    if len(neural_trial[str(trial)]['time']) < length:
        length = -1
    fluo = neural_trial[str(trial)][ch][0:num, start:start+length]
    time = neural_trial[str(trial)]['time'][start:start+length]
    stim = neural_trial[str(trial)]['stim'][start:start+length]
    fig, axs = plt.subplots(1, figsize=(10, 8))
    stim_color = ['white', 'dodgerblue']
    # plot stimulus.
    for i in range(len(stim)-1):
        axs.fill_between(
            [time[i], time[i+1]], -1, num,
            color=stim_color[int(stim[i])])
    # plot traces.
    for i in range(fluo.shape[0]):
        axs.plot(time, fluo[i,:]+i, color='black')
    # adjust layout.
    axs.tick_params(tick1On=True)
    axs.spines['left'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('time / ms')
    axs.set_ylabel('neuron id')
    axs.set_xlim([time[0], time[-1]])
    axs.set_ylim([-1, num])
    axs.set_yticks(np.arange(num))
    axs.yaxis.grid(True)
    fig.suptitle('Example traces for {} neurons'.format(num))
    fig.tight_layout()
    # save figure.
    fig.savefig(os.path.join(ops['save_path0'], 'fig2_traces.png'), dpi=300)
    plt.close()
