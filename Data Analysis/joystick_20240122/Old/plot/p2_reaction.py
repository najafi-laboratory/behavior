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


def plot_curves(axs, session_data, early_idx, mid_idx, late_idx, max_time=7):
    reaction = session_data['reaction']
    reaction_early = [reaction[i] for i in early_idx]
    reaction_mid = [reaction[i] for i in mid_idx]
    reaction_late = [reaction[i] for i in late_idx]
    reaction = [reaction_early, reaction_mid, reaction_late]
    cmap = ['royalblue', 'green', 'coral']
    label = ['Early', 'Mid', 'Late']
    group_reaction = [np.concatenate(r, axis=0) for r in reaction_late]
    group_reaction = np.concatenate(group_reaction, axis=0)
    bin_stat = bin_trials(group_reaction, max_time)
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
        [0, 1.8], 1, 0.25,
        color='gold', alpha=0.2,
        label='Pre-perturb stim')
    axs.fill_between(
        [1.8, 3.8], 1, 0.25,
        color='coral', alpha=0.2,
        label='Post-perturb stim')
    axs.set_xlim([0, max_time])
    axs.set_ylim([0.20, 1.05])
    axs.set_xlabel('Reaction time (since stim onset) / s')
    axs.set_ylabel('Probability of correct choice')
    axs.set_xticks(np.arange(0, max_time, 1))
    axs.set_yticks([0.25, 0.50, 0.75, 1])
    #axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)


def plot_p2(
    session_data_1,
    session_data_2,
    ):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    plt.subplots_adjust(hspace=0.7)
    plot_curves(
        axs[0], session_data_1,
        early_idx=[64, 65, 72],
        mid_idx=[52, 54, 55, 57],
        late_idx=[51, 52, 54, 55, 57, 59, 60, 62, 63, 66, 70, 73, 76])
    plot_curves(
        axs[1], session_data_2,
        early_idx=[22, 30, 31],
        mid_idx=[33, 37, 45],
        late_idx=[25, 37, 38, 41, 42, 44, 45])
    axs[0].set_title('Mouse 1')
    axs[1].set_title('Mouse 2')
    fig.suptitle('Performance vs. Reaction time (the 1st side lick since stim onset)')
    fig.tight_layout()
    print('Plot p2 completed.')
    fig.savefig('./figures/p2_reaction.pdf', dpi=300)
    fig.savefig('./figures/p2_reaction.png', dpi=300)