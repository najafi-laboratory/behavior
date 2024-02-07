import numpy as np
import matplotlib.pyplot as plt


def bin_trials(trial_reaction, max_time, bin_size=0.5, least_trials=2):
    num_bins = int(max_time/bin_size)
    bin_stat = []
    for i in range(num_bins):
        center = i*bin_size
        idx = np.where(np.abs(trial_reaction[:,0]-center)<bin_size/2)[0]
        if len(idx) > least_trials:
            bin_stat.append([
                center, np.sum(trial_reaction[idx,1])/len(idx)])
    bin_stat = np.array(bin_stat).reshape(-1, 2)
    return bin_stat


def plot_subject(
        ax,
        subject_session_data,
        max_sessions=5,
        max_time=7
        ):
    subject = subject_session_data['subject']
    date = subject_session_data['dates'][-1]
    reaction = subject_session_data['reaction'][-1]
    reaction = np.concatenate(reaction, axis=0)
    bin_stat = bin_trials(reaction, max_time)
    ax.plot(
        bin_stat[:,0], bin_stat[:,1],
        color='black',
        label='Reaction time')
    ax.scatter(
        bin_stat[:,0], bin_stat[:,1],
        color='black',
        s=5)
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.fill_between(
        [0, 1.3], 1, 0.25,
        color='gold', alpha=0.2,
        label='Pre-perturb stim')
    ax.fill_between(
        [1.3, 4.3], 1, 0.25,
        color='coral', alpha=0.2,
        label='Post-perturb stim')
    ax.set_xlim([0, max_time])
    ax.set_ylim([0.20, 1.05])
    ax.set_xlabel('Reaction time (since stim onset) / s')
    ax.set_ylabel('Probability of correct choice')
    ax.set_xticks(np.arange(0, max_time, 1))
    ax.set_yticks([0.25, 0.50, 0.75, 1])
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    ax.set_title(
        subject + ' ' + date +
        ' Performance vs. Reaction time (the 1st side lick since stim onset)')
    

def plot_fig8(
        session_data,
        max_sessions=5,
        max_time=7
        ):
    fig, axs = plt.subplots(
        len(session_data), 1,
        figsize=(16, 8*len(session_data)))
    plt.subplots_adjust(hspace=2)
    for i in range(len(session_data)):
        plot_subject(
            axs[i],
            session_data[i],
            max_sessions=max_sessions,
            max_time=max_time)
    print('Completed fig8')
    fig.set_size_inches(10, len(session_data)*3)
    fig.tight_layout()
    fig.savefig('./figures/fig8_reaction_trial.pdf', dpi=300)
    fig.savefig('./figures/fig8_reaction_trial.png', dpi=300)
    plt.close()
    