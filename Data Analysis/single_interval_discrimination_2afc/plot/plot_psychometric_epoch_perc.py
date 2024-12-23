import numpy as np
from scipy.stats import sem

def get_bin_stat(decision, isi='post'):
    bin_size=100
    least_trials=3
    bins = np.arange(0, 1000 + bin_size, bin_size)
    bins = bins - bin_size / 2
    bin_indices = np.digitize(decision[3,:], bins) - 1
    bin_mean = []
    bin_sem = []
    for i in range(len(bins)-1):
        direction = decision[1, bin_indices == i].copy()
        m = np.mean(direction) if len(direction) > least_trials else np.nan
        s = sem(direction) if len(direction) > least_trials else np.nan
        bin_mean.append(m)
        bin_sem.append(s)
    bin_mean = np.array(bin_mean)
    bin_sem  = np.array(bin_sem)
    bin_isi  = bins[:-1] + (bins[1]-bins[0]) / 2
    non_nan  = (1-np.isnan(bin_mean)).astype('bool')
    bin_mean = bin_mean[non_nan]
    bin_sem  = bin_sem[non_nan]
    bin_isi  = bin_isi[non_nan]
    return bin_mean, bin_sem, bin_isi

def plot_curves(ax, subject, jitter_session, dates, decision, isi_post_perc):
    decision = np.concatenate([decision, isi_post_perc], axis=0)
    non_nan = (1-np.isnan(np.sum(decision, axis=0))).astype('bool')
    decision = decision[:,non_nan]
    bin_mean, bin_sem, bin_isi = get_bin_stat(decision)
    ax.plot(
        bin_isi, bin_mean,
        linestyle='-' if jitter_session==0 else '--',
        label=dates[4:],
        marker='.',
        markersize=4)
    ax.hlines(0.5, 0.0, 1000, linestyle=':', color='grey')
    ax.vlines(500, 0.0, 1.0, linestyle=':', color='grey')
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([-50,1050])
    ax.set_ylim([-0.05,1.05])
    ax.set_xticks(np.arange(6)*200)
    ax.set_yticks(np.arange(5)*0.25)
    ax.set_xlabel('post perturbation isi')
    ax.set_ylabel('right fraction')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    ax.set_title('psychometric function (percetual mean)')

def run(axs, subject_session_data):
    max_sessions = 6
    subject = subject_session_data['subject']
    dates = subject_session_data['dates']
    stim_seq = subject_session_data['stim_seq']
    decision = subject_session_data['decision']
    for s in range(len(stim_seq)):
        for t in range(len(stim_seq[s])):
            if stim_seq[s][t].shape[1] < 4:
                decision[s][t] = np.array([[np.nan], [np.nan], [np.nan]])
            elif decision[s][t][0,0] < stim_seq[s][t][0,3]:
                decision[s][t] = np.array([[np.nan], [np.nan], [np.nan]])
    decision = [np.concatenate(d, axis=1) for d in decision]
    isi_post_perc = subject_session_data['isi_post_perc']
    isi_post_perc = [np.array(isi).reshape(1,-1) for isi in isi_post_perc]
    jitter_flag = subject_session_data['jitter_flag']
    if len(axs)*max_sessions <= len(dates):
        dates = dates[-len(axs)*max_sessions:]
        decision = decision[-len(axs)*max_sessions:]
        isi_post_perc = isi_post_perc[-len(axs)*max_sessions:]
        jitter_flag = jitter_flag[-len(axs)*max_sessions:]
    jitter_session = np.array([np.sum(j) for j in jitter_flag])
    jitter_session[jitter_session!=0] = 1
    for i in range(len(dates)):
        plot_curves(
            axs[i//max_sessions], subject,
            jitter_session[i], dates[i], decision[i], isi_post_perc[i])
