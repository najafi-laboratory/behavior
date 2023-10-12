import numpy as np
import matplotlib.pyplot as plt


def plot_scatter(axs, session_data):
    subject = session_data['subject']
    isi = session_data['isi'][session_data['LR12_start']:]
    dates = session_data['dates'][session_data['LR12_start']:]
    for i in range(len(isi)):
        if len(isi[i]) > 0:
            duration = np.concatenate(isi[i])
        else:
            duration = isi[i]
        axs.scatter(
            np.zeros_like(duration) + i + 1, duration,
            color='dodgerblue',
            alpha=0.2,
            s=5)
    axs.tick_params(tick1On=False)
    axs.spines['left'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_ylim([0, 1.0])
    axs.set_xlabel('training session')
    axs.set_ylabel('isi / s')
    axs.set_xticks(np.arange(len(isi))+1)
    axs.set_xticklabels(dates, rotation='vertical')
    axs.set_yticks(np.arange(11)*0.1)
    axs.yaxis.grid(True)
    axs.set_title(subject)


def plot_fig5(
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
    fig.suptitle('stimulus isi across sessions')
    fig.tight_layout()
    print('Plot fig5 completed.')
    fig.savefig('./figures/fig5_stim_isi.pdf', dpi=300)
    fig.savefig('./figures/fig5_stim_isi.png', dpi=300)
