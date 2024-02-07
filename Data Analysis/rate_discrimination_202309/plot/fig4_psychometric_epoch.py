import numpy as np
import matplotlib.pyplot as plt


def bin_trials(trial_choice, bin_size=100, least_trials=3):
    num_bins = int(1000/bin_size)
    bin_stat = []
    for i in range(num_bins):
        center = i*bin_size
        idx = np.where(np.abs(trial_choice[:,0]*1000-center)<bin_size/2)[0]
        if len(idx) > least_trials:
            bin_stat.append([
                center/1000,
                np.mean(trial_choice[idx,1]),
                np.std(trial_choice[idx,1])])
    bin_stat = np.array(bin_stat).reshape(-1, 3)
    return bin_stat


def plot_curves(axs, subject, dates, choice):
    cmap = plt.cm.plasma_r(np.arange(len(choice))/len(choice))
    for i in range(len(choice)):
        if len(choice[i]) > 0:
            trial_choice = np.concatenate(choice[i]).reshape(-1,2)
            bin_stat = bin_trials(trial_choice)
            axs.plot(
                bin_stat[:,0], bin_stat[:,1],
                color=cmap[i],
                label=dates[i][4:])
            axs.scatter(
                bin_stat[:,0], bin_stat[:,1],
                color=cmap[i], s=5)
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
    axs.set_ylabel('right fraction')
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)


def plot_subject(
        subject_session_data,
        max_subplots,
        max_sessions,
        ):
    subject = subject_session_data['subject']
    dates = subject_session_data['dates'][subject_session_data['LR12_start']:]
    choice = subject_session_data['choice'][subject_session_data['LR12_start']:]
    if len(np.concatenate(choice)) > 0:
        if len(dates) <= max_sessions:
            max_subplots = 1
            start_idx = 0
        else:
            max_subplots = min(int(len(dates)/max_sessions), max_subplots)
            start_idx = len(dates) - max_subplots * max_sessions
        dates = dates[start_idx:]
        choice = choice[start_idx:]
        fig, axs = plt.subplots(max_subplots, 1, figsize=(4, max_subplots*2))
        plt.subplots_adjust(hspace=0.4)
        plt.subplots_adjust(wspace=0.4)
        if max_subplots > 1:
            for i in range(max_subplots):
                plot_curves(
                    axs[i], subject,
                    dates[i*max_sessions:(i+1)*max_sessions],
                    choice[i*max_sessions:(i+1)*max_sessions])
        else:
            plot_curves(
                axs, subject,
                dates,
                choice)
        fig.suptitle(subject + ' psychometric functions')
        fig.set_size_inches(4, len(subject_session_data)*4)
        fig.tight_layout()
        print('Completed fig4 for ' + subject)
        fig.savefig('./figures/fig4_psychometric_epoch_'+subject+'.pdf', dpi=300)
        fig.savefig('./figures/fig4_psychometric_epoch_'+subject+'.png', dpi=300)
        plt.close()
    else:
        print('Plot fig4 failed. Found no non-naive trials for ' + subject)
    

def plot_fig4(
        session_data,
        max_subplots=5,
        max_sessions=6
        ):
    for i in range(len(session_data)):
        plot_subject(
            session_data[i],
            max_subplots=max_subplots,
            max_sessions=max_sessions)
    print('Completed fig4')