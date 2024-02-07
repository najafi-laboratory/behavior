import numpy as np
import matplotlib.pyplot as plt


def plot_subject(
        ax,
        subject_session_data,
        max_sessions=25
        ):
    subject = subject_session_data['subject']
    duration = subject_session_data['avsync'][subject_session_data['LR12_start']:]
    dates = subject_session_data['dates'][subject_session_data['LR12_start']:]
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    duration = duration[start_idx:]
    dates = dates[start_idx:]
    for i in range(len(duration)):
        ax.scatter(
            np.zeros_like(duration[i]) + i + 1, duration[i],
            color='dodgerblue',
            alpha=0.2,
            s=5)
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([-0.2,1.0])
    ax.set_xlabel('training session')
    ax.set_ylabel('vis to aud start / s')
    ax.set_xticks(np.arange(len(duration))+1)
    ax.set_xticklabels(dates, rotation='vertical')
    ax.set_yticks(np.arange(-2, 11, 1)*0.1)
    ax.yaxis.grid(True)
    ax.set_title(subject + ' vis to audio start across sessions')
    
    
def plot_fig6(
        session_data,
        max_sessions=25
        ):
    fig, axs = plt.subplots(
        len(session_data), 1,
        figsize=(16, 8*len(session_data)))
    plt.subplots_adjust(hspace=2)
    for i in range(len(session_data)):
        plot_subject(
            axs[i],
            session_data[i],
            max_sessions=max_sessions)
    print('Completed fig6')
    fig.set_size_inches(12, len(session_data)*4)
    fig.tight_layout()
    fig.savefig('./figures/fig6_av_sync.pdf', dpi=300)
    fig.savefig('./figures/fig6_av_sync.png', dpi=300)
    plt.close()