import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

def get_bin_stat(decision, isi='post'):
    # bin_size=100
    bin_size=7
    least_trials=5
    # set bins across isi range
    # short ISI: [50, 400, 750]ms.  associated with left lick
    # long ISI: [750, 1100, 1450]ms.  associated with right lick
    # bins = np.arange(0, 1000 + bin_size, bin_size)
    bins = np.arange(0, 1500 + bin_size, bin_size)
    bins = bins - bin_size / 2
    bin_indices = np.digitize(decision[3,:], bins) - 1
    bin_mean = []
    bin_sem = []
    trials_per_bin = []    
    for i in range(len(bins)-1):
        direction = decision[1, bin_indices == i].copy()
        m = np.mean(direction) if len(direction) > 0 else np.nan
        s = sem(direction) if len(direction) > least_trials else np.nan
        num_trials = len(bin_indices[bin_indices == i]) if len(direction) > 0 else np.nan
        trials_per_bin.append(num_trials)         
        bin_mean.append(m)
        bin_sem.append(s)
    bin_mean = np.array(bin_mean)
    bin_sem  = np.array(bin_sem)
    bin_isi  = bins[:-1] + (bins[1]-bins[0]) / 2
    non_nan  = (1-np.isnan(bin_mean)).astype('bool')
    bin_mean = bin_mean[non_nan]
    bin_sem  = bin_sem[non_nan]
    bin_isi  = bin_isi[non_nan]
    trials_per_bin = np.array(trials_per_bin)
    trials_per_bin = trials_per_bin[non_nan]      
    return bin_mean, bin_sem, bin_isi, trials_per_bin

def plot_curves(ax, subject,jitter_session, dates, decision, post_isi_mean,j,r, k ,n_jitter,n_control , n_chemo, chemo_labels):
    decision = np.concatenate([decision, post_isi_mean], axis=0)
    non_nan = (1-np.isnan(np.sum(decision, axis=0))).astype('bool')
    # ax.hlines(0.5, 0.0, 1000, linestyle='--' , color='silver' , lw = 0.5)
    # ax.hlines(0.5, 0.0, 1600, linestyle='--' , color='silver' , lw = 0.5)
    # ax.vlines(500, 0.0, 1.0, linestyle='--' , color='silver', lw = 0.5)    
    extra = 1
    decision = decision[:,non_nan]
    bin_mean, bin_sem, bin_isi, trials_per_bin = get_bin_stat(decision)
    c1 = (n_control+extra-r)/(n_control+extra)
    c2 = (n_jitter+extra-j)/(n_jitter+extra)
    c3 = (n_chemo+extra-k)/(n_chemo+extra)
    if chemo_labels == 1:
        c = [1 , c3 , c3]
    elif jitter_session == 1:
        c = [c2 , 1 , c2]
    else:
        c = [c1 , c1 , 1]
    # ax.plot(
    #     bin_isi, bin_mean,
    #     label=dates[4:],
    #     marker='.',
    #     markersize=4,
    #     color = c)
    ax.plot(
        bin_isi, trials_per_bin,
        label=dates[4:],
        marker='.',
        markersize=4,
        color = c)    
    ax.tick_params(tick1On=False)
    ax.tick_params(axis='x', rotation=45)    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_xlim([-50,1050])
    ax.set_xlim([-50,1600])
    ax.vlines(750, 0.0, np.max(trials_per_bin) + 10, linestyle='--' , color='silver', lw = 0.5)    
    ax.set_ylim([0,np.max(trials_per_bin) + 10])
    # ax.set_xticks(np.arange(6)*200)
    ax.set_xticks(np.arange(11)*150)
    # ax.set_yticks(np.arange(5)*0.25)
    ax.set_xlabel('post perturbation isi')
    ax.set_ylabel('number of trials')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    ax.set_title('single session isi distribution')

def run(axs, subject_session_data):
    max_sessions = 5
    subject = subject_session_data['subject']
    dates = subject_session_data['dates']
    chemo_labels = subject_session_data['Chemo']
    
    decision = subject_session_data['decision']
    decision = [np.concatenate(d, axis=1) for d in decision]
    isi_post_emp = subject_session_data['isi_post_emp']
    isi_post_emp = [np.array(isi).reshape(1,-1) for isi in isi_post_emp]
    
    jitter_flag = subject_session_data['jitter_flag']
    if len(axs)*max_sessions <= len(dates):
        dates = dates[-len(axs)*max_sessions:]
        decision = decision[-len(axs)*max_sessions:]
        isi_post_emp = isi_post_emp[-len(axs)*max_sessions:]
        jitter_flag = jitter_flag[-len(axs)*max_sessions:]
        chemo_labels = chemo_labels[-len(axs)*max_sessions:]
    jitter_session = np.array([np.sum(j) for j in jitter_flag])
    jitter_session[jitter_session!=0] = 1
    r = 0
    j = 0
    k = 0
    n_jitter = []
    n_control = []
    n_chemo = []
    for i in range(len(dates)//max_sessions+1):
        a = jitter_session[i*max_sessions:min(len(dates) , (i+1)*max_sessions)]
        b = chemo_labels[i*max_sessions:min(len(dates) , (i+1)*max_sessions)]
        n_chemo1 = np.count_nonzero(b)
        n_jitter1 = np.count_nonzero(a)
        if len(a) - n_jitter1-n_chemo1 >-1:
            n_control1 = len(a) - n_jitter1-n_chemo1
        else:
            n_control1 = 0
        n_jitter.append(n_jitter1)
        n_chemo.append(n_chemo1)
        n_control.append(n_control1)
    for i in range(len(dates)):
        if i%max_sessions != 0:
            if jitter_session[i] == 1:
                j = j + 1
            elif chemo_labels[i] == 1:
                k = k + 1
            else:
                r = r + 1
        else:
            j = 0
            r = 0
            k = 0
            if jitter_session[i] == 1:
                j = j + 1
            elif chemo_labels[i] == 1:
                k = k + 1
            else:
                r = r + 1
        plot_curves(
            axs[i//max_sessions], subject,
            jitter_session[i], dates[i], decision[i], isi_post_emp[i], j, r, k ,n_jitter[i//max_sessions],n_control[i//max_sessions] , n_chemo[i//max_sessions] , chemo_labels[i])
