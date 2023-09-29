import numpy as np
import matplotlib.pyplot as plt


def bin_trials(trial_choice, bin_size):
    num_bins = int(1000/bin_size)
    bin_stat = []
    for i in range(num_bins):
        center = i*bin_size + bin_size/2
        idx = np.where(np.abs(trial_choice[:,0]*1000-center)<bin_size/2)[0]
        if len(idx) > 0:
            bin_stat.append([
                center/1000,
                np.mean(trial_choice[idx,1]),
                np.std(trial_choice[idx,1])])
    bin_stat = np.array(bin_stat).reshape(-1, 3)
    return bin_stat


def plot_curves(fig, axs, session_data, bin_size=100):
    subject = session_data['subject']
    dates = session_data['dates'][session_data['LR12_start']:]
    choice = session_data['choice'][session_data['LR12_start']:]
    cmap = plt.cm.hot_r(np.arange(len(choice))/len(choice))
    for i in range(len(choice)):
        trial_choice = np.concatenate(choice[i]).reshape(-1,2)
        bin_stat = bin_trials(trial_choice, bin_size=bin_size)
        axs.plot(
            bin_stat[:,0], bin_stat[:,1],
            color=cmap[i])
        '''
        axs.errorbar(
            x = bin_stat[:,0],
            y = bin_stat[:,1],
            yerr = bin_stat[:,2],
            color=cmap[i], lw=1, marker='P', ms=8, mec='w', mew='1')
        '''
    axs.tick_params(tick1On=False)
    axs.spines['left'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlim([0.0,1.0])
    axs.set_ylim([0.0,1.0])
    axs.set_xticks(np.arange(11)*0.1)
    axs.set_yticks(np.arange(5)*0.25)
    axs.set_xlabel('isi')
    axs.set_ylabel('right choice fraction')
    axs.xaxis.grid(True)
    axs.yaxis.grid(True)
    axs.set_title(subject)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=plt.cm.hot_r),
        ax=axs,
        label='session id',
        ticks=[0, 1])
    cbar.ax.set_yticklabels(
        [dates[0], dates[-1]])


def plot_fig4(
    session_data_1,
    session_data_2,
    session_data_3,
    session_data_4
    ):
    fig, axs = plt.subplots(2, 2, figsize=(20, 8))
    plt.subplots_adjust(hspace=0.7)
    plot_curves(fig, axs[0,0], session_data_1)
    plot_curves(fig, axs[0,1], session_data_2)
    plot_curves(fig, axs[1,0], session_data_3)
    plot_curves(fig, axs[1,1], session_data_4)
    fig.suptitle('psychometric functions')
    fig.tight_layout()
    fig.savefig('./figures/fig4_psychometric_epoch.pdf', dpi=300)
    fig.savefig('./figures/fig4_psychometric_epoch.png', dpi=300)

