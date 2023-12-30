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


def plot_curves(axs, session_data):
    subject = session_data['subject']
    dates = session_data['dates']
    com = session_data['com']
    dates_list, com_list = check_null_list(dates, com)
    right2left, left2right = get_com_percentage(com_list)
    axs.barh(
        dates_list, right2left,
        left=0,
        edgecolor='white',
        height=0.5,
        color='dodgerblue',
        label='right to left')
    axs.barh(
        dates_list, left2right,
        left=right2left,
        edgecolor='white',
        height=0.5,
        color='hotpink',
        label='left to right')
    axs.vlines(
        0.5, -1, len(dates_list),
        linestyle=':', color='grey')
    axs.tick_params(tick1On=False)
    axs.spines['top'].set_visible(False)
    axs.set_xlim([0, 1])
    axs.set_ylim([-1, len(dates_list)])
    axs.set_xlabel('Percentage')
    axs.set_ylabel('Date')
    axs.set_yticks(np.arange(0,len(dates_list)))
    axs.set_yticklabels(dates_list)
    axs.set_title(subject + ' ')
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)


def plot_fig10(
    session_data_1,
    session_data_2,
    session_data_3,
    session_data_4,
    ):
    fig, axs = plt.subplots(2, 2, figsize=(9, 15))
    plt.subplots_adjust(hspace=0.7)
    plot_curves(axs[0,0], session_data_1)
    plot_curves(axs[0,1], session_data_2)
    plot_curves(axs[1,0], session_data_3)
    plot_curves(axs[1,1], session_data_4)
    fig.suptitle('Change of mind percentage across sessions')
    fig.tight_layout()
    print('Plot fig10 completed.')
    fig.savefig('./figures/fig10_com.pdf', dpi=300)
    fig.savefig('./figures/fig10_com.png', dpi=300)