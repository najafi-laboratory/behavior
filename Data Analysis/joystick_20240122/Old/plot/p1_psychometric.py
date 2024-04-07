import numpy as np
import matplotlib.pyplot as plt


def bin_trials(trial_choice, bin_size=100, least_trials=2):
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


def plot_curves(axs, choice, color, label):
    bin_stat = bin_trials(choice)
    axs.plot(
        bin_stat[:,0], bin_stat[:,1],
        color=color,
        label=label)
    axs.scatter(
        bin_stat[:,0], bin_stat[:,1],
        color=color, s=1)
    axs.hlines(
        0.5, 0.0, 1.0,
        lw=0.5,
        linestyle='--', color='grey')
    axs.vlines(
        0.5, 0.0, 1.0,
        lw=0.5,
        linestyle='--', color='grey')


def plot_subject_psychometric(axs, session_data, early_idx, mid_idx, late_idx):
    subject = session_data['subject']
    choice = session_data['choice']
    choice_early = [choice[i] for i in early_idx]
    choice_mid = [choice[i] for i in mid_idx]
    choice_late = [choice[i] for i in late_idx]
    choice_early = [np.concatenate(choice_early[i]).reshape(-1,2)
                    for i in range(len(early_idx))]
    choice_mid = [np.concatenate(choice_mid[i]).reshape(-1,2)
                  for i in range(len(mid_idx))]
    choice_late = [np.concatenate(choice_late[i]).reshape(-1,2)
                   for i in range(len(late_idx))]
    choice = [choice_early, choice_mid, choice_late]
    label = ['Early', 'Mid', 'Late']
    cmap = ['royalblue', 'green', 'coral']
    for i in range(3):
        c = np.concatenate(choice[i])
        plot_curves(axs, c, cmap[i], label[i])
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlim([-0.05,1.05])
    axs.set_ylim([-0.05,1.05])
    axs.set_xticks(np.arange(6)*0.2)
    axs.set_yticks(np.arange(5)*0.25)
    axs.set_xlabel('Post-perturbation ISI')
    axs.set_ylabel('Probability of choosing right side')
    axs.set_title(subject)
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)


def plot_p1(
    session_data_1,
    session_data_2,
    ):
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    plot_subject_psychometric(
        axs[0], session_data_1,
        early_idx=[54, 66, 78],
        mid_idx=[50, 53, 64],
        late_idx=[66, 70, 73])
    plot_subject_psychometric(
        axs[1], session_data_2,
        early_idx=[11, 16, 30, 31],
        mid_idx=[33, 37, 45],
        late_idx=[25, 40, 41, 42, 43])
    axs[0].set_title('Mouse 1')
    axs[1].set_title('Mouse 2')
    fig.suptitle('Psychometric function')
    fig.tight_layout()
    print('Plot p1 completed.')
    fig.savefig('./figures/p1_psychometric.pdf', dpi=300)
    fig.savefig('./figures/p1_psychometric.png', dpi=300)