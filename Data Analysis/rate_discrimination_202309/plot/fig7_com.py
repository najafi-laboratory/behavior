import numpy as np
import matplotlib.pyplot as plt


def check_null_list(dates, com):
    dates_list = []
    com_list = []
    for i in range(len(dates)):
        if len(com[i]) > 0:
            dates_list.append(dates[i])
            com_list.append(com[i])
    return dates_list, com_list


def get_com_percentage(com_list):
    left2right = np.array([np.mean(com) for com in com_list])
    right2left = 1 - left2right
    return right2left, left2right


def plot_subject(
        ax,
        subject_session_data
        ):
    subject = subject_session_data['subject']
    dates = subject_session_data['dates']
    com = subject_session_data['com']
    if len(np.concatenate(com)) > 0:
        dates_list, com_list = check_null_list(dates, com)
        right2left, left2right = get_com_percentage(com_list)
        ax.barh(
            dates_list, right2left,
            left=0,
            edgecolor='white',
            height=0.5,
            color='dodgerblue',
            label='right to left')
        ax.barh(
            dates_list, left2right,
            left=right2left,
            edgecolor='white',
            height=0.5,
            color='hotpink',
            label='left to right')
        ax.vlines(
            0.5, -1, len(dates_list),
            linestyle=':', color='grey')
        ax.tick_params(tick1On=False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([0, 1])
        ax.set_ylim([-1, len(dates_list)])
        ax.set_xlabel('Percentage')
        ax.set_ylabel('Date')
        ax.set_yticks(np.arange(0,len(dates_list)))
        ax.set_yticklabels(dates_list)
        ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
        ax.set_title(subject + ' Change of mind percentage')
    else:
        print('Plot fig7 failed. Found no change of mind trials for ' + subject)
    
    
def plot_fig7(
        session_data,
        ):
    fig, axs = plt.subplots(
        1, len(session_data),
        figsize=(16, 8*len(session_data)))
    plt.subplots_adjust(hspace=2)
    for i in range(len(session_data)):
        plot_subject(
            axs[i],
            session_data[i])
    print('Completed fig7')
    fig.set_size_inches(len(session_data)*4, 12)
    fig.tight_layout()
    fig.savefig('./figures/fig7_com.pdf', dpi=300)
    fig.savefig('./figures/fig7_com.png', dpi=300)
    plt.close()