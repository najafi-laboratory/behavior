import numpy as np
import matplotlib.pyplot as plt


def plot_fig5(
    session_data,
    max_sessions=25
    ):
    fig, axs = plt.subplots(1, figsize=(10, 4))
    subject = session_data['subject']
    isi = session_data['isi'][session_data['LR12_start']:]
    dates = session_data['dates'][session_data['LR12_start']:]
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    isi = isi[start_idx:]
    dates = dates[start_idx:]
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
    fig.suptitle(subject + ' stimulus isi across sessions')
    fig.tight_layout()
    print('Completed fig5 for ' + subject)
    fig.savefig('./figures/fig5_'+subject+'_stim_isi.pdf', dpi=300)
    fig.savefig('./figures/fig5_'+subject+'_stim_isi.png', dpi=300)
    plt.close()
    
    