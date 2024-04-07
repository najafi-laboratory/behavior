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


def plot_fig8(
    session_data,
    max_sessions=5,
    max_time=7
    ):
    fig, axs = plt.subplots(1, figsize=(8, 4))
    subject = session_data['subject']
    date = session_data['dates'][-1]
    reaction = session_data['reaction'][-1]
    reaction = np.concatenate(reaction, axis=0)
    bin_stat = bin_trials(reaction, max_time)
    axs.plot(
        bin_stat[:,0], bin_stat[:,1],
        color='black',
        label='Reaction time')
    axs.scatter(
        bin_stat[:,0], bin_stat[:,1],
        color='black',
        s=5)
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.fill_between(
        [0, 1.3], 1, 0.25,
        color='gold', alpha=0.2,
        label='Pre-perturb stim')
    axs.fill_between(
        [1.3, 4.3], 1, 0.25,
        color='coral', alpha=0.2,
        label='Post-perturb stim')
    axs.set_xlim([0, max_time])
    axs.set_ylim([0.20, 1.05])
    axs.set_xlabel('Reaction time (since stim onset) / s')
    axs.set_ylabel('Probability of correct choice')
    axs.set_xticks(np.arange(0, max_time, 1))
    axs.set_yticks([0.25, 0.50, 0.75, 1])
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs.set_title(subject + ' ' + date)
    fig.suptitle('Performance vs. Reaction time (the 1st side lick since stim onset)')
    fig.tight_layout()
    print('Completed fig8 for ' + subject)
    fig.savefig('./figures/fig8_'+subject+'_reaction_trial.pdf', dpi=300)
    fig.savefig('./figures/fig8_'+subject+'_reaction_trial.png', dpi=300)
    plt.close()
    
    
    