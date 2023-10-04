import numpy as np
import matplotlib.pyplot as plt


def plot_scatter(axs, session_data):
    subject = session_data['subject']
    duration = session_data['avsync'][session_data['LR12_start']:]
    dates = session_data['dates'][session_data['LR12_start']:]
    for i in range(len(duration)):
        axs.scatter(
            np.zeros_like(duration[i]) + i + 1, duration[i],
            color='dodgerblue',
            alpha=0.2,
            s=5)
    axs.tick_params(tick1On=False)
    axs.spines['left'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_ylim([-0.2,1.0])
    axs.set_xlabel('training session')
    axs.set_ylabel('vis to aud start / s')
    axs.set_xticks(np.arange(len(duration))+1)
    axs.set_xticklabels(dates, rotation='vertical')
    axs.set_yticks(np.arange(-2, 11, 1)*0.1)
    axs.yaxis.grid(True)
    axs.set_title(subject)


def plot_fig6(
    session_data_1,
    session_data_2,
    session_data_3,
    session_data_4
    ):
    fig, axs = plt.subplots(2, 2, figsize=(20, 8))
    plt.subplots_adjust(hspace=0.7)
    plot_scatter(axs[0,0], session_data_1)
    plot_scatter(axs[0,1], session_data_2)
    plot_scatter(axs[1,0], session_data_3)
    plot_scatter(axs[1,1], session_data_4)
    fig.suptitle('vis to audio start across sessions')
    fig.tight_layout()
    fig.savefig('./figures/fig6_av_sync.pdf', dpi=300)
    fig.savefig('./figures/fig6_av_sync.png', dpi=300)
