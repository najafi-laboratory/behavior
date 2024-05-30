import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem


# bin the data with timestamps.

def get_bin_stat(choice, bin_size=0.1, min_samples=3):
    bins = np.arange(0, 1 + bin_size, bin_size)
    bins = bins - bin_size / 2
    bin_indices = np.digitize(choice[:,0], bins) - 1
    bin_mean = []
    bin_sem = []
    for i in range(len(bins)-1):
        bin_values = choice[bin_indices == i,1]
        m = np.mean(bin_values) if len(bin_values) > min_samples else np.nan
        s = sem(bin_values) if len(bin_values) > min_samples else np.nan
        bin_mean.append(m)
        bin_sem.append(s)
    bin_mean = np.array(bin_mean)
    bin_sem = np.array(bin_sem)
    return bins, bin_mean, bin_sem


def plot_subject(subject_session_data):
    subject = subject_session_data['subject']
    choice = subject_session_data['choice']
    choice = [c for c in choice if len(c)>0]
    choice = np.concatenate(choice)
    bins, bin_mean, bin_sem = get_bin_stat(choice)
    fig, axs = plt.subplots(1, 1, figsize=(4, 3))
    axs.plot(
        bins[:-1] + (bins[1]-bins[0]) / 2,
        bin_mean,
        color='dodgerblue',
        marker='.',
        label='fix',
        markersize=4)
    axs.fill_between(
        bins[:-1] + (bins[1]-bins[0]) / 2,
        bin_mean - bin_sem,
        bin_mean + bin_sem,
        color='dodgerblue',
        alpha=0.2)
    axs.hlines(
        0.5, 0.0, 1.0,
        linestyle=':', color='grey')
    axs.vlines(
        0.5, 0.0, 1.0,
        linestyle=':', color='grey')
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlim([-0.05,1.05])
    axs.set_ylim([-0.05,1.05])
    axs.set_xticks(np.arange(6)*0.2)
    axs.set_yticks(np.arange(5)*0.25)
    axs.set_xlabel('isi')
    axs.set_ylabel('prob. of choosing the right side (mean$\pm$sem)')
    fig.suptitle(subject + ' psychometric functions grand average')
    fig.set_size_inches(5, 4)
    fig.tight_layout()
    fig.savefig('./figures/fig1_psychometric_'+subject+'.pdf', dpi=300)
    fig.savefig('./figures/fig1_psychometric_'+subject+'.png', dpi=300)
    plt.close()


def plot_fig1(session_data):
    for i in range(len(session_data)):
        plot_subject(session_data[i])
    print('Completed fig1')
