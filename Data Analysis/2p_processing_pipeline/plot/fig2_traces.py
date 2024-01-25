#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from modules import RetrieveResults


save_folder = 'FN8_PPC_011724'


def plot_fig2(
        ch = 'fluo_ch2',
        trial = 1,
        num = 20,
        idx = 0,
        start = 1000,
        length = 1000,
        amp = 1,
        ):
    [ops, neural_trial, mask] = RetrieveResults.run(save_folder)
    fluo = neural_trial[str(trial)][ch][idx:idx+num, start:start+length]
    time = neural_trial[str(trial)]['time'][start:start+length]
    stim = neural_trial[str(trial)]['stim'][start:start+length]
    fig, axs = plt.subplots(1, figsize=(10, 8))
    stim_color = ['white', 'dodgerblue']
    for i in range(len(stim)-1):
        axs.fill_between(
            [time[i], time[i+1]], -1, num,
            color=stim_color[int(stim[i])])
    for i in range(fluo.shape[0]):
        axs.plot(time, fluo[i,:]*amp+i, color='black')
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
    fig.suptitle('Traces for {} neurons'.format(num))
    fig.tight_layout()
    fig.savefig('./figures/fig2_traces.pdf', dpi=300)
    fig.savefig('./figures/fig2_traces.png', dpi=300)


plot_fig2()