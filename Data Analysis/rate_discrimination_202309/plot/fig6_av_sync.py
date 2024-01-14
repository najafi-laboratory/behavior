import numpy as np
import matplotlib.pyplot as plt


def plot_fig6(
    session_data,
    max_sessions=25
    ):
    fig, axs = plt.subplots(1, figsize=(10, 4))
    subject = session_data['subject']
    duration = session_data['avsync'][session_data['LR12_start']:]
    dates = session_data['dates'][session_data['LR12_start']:]
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    duration = duration[start_idx:]
    dates = dates[start_idx:]
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
    fig.suptitle('vis to audio start across sessions')
    fig.tight_layout()
    print('Completed fig6 for ' + subject)
    fig.savefig('./figures/fig6_'+subject+'_av_sync.pdf', dpi=300)
    fig.savefig('./figures/fig6_'+subject+'_av_sync.png', dpi=300)
    plt.close()