import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def save_image(filename): 
    
    # PdfPages is a wrapper around pdf  
    # file so there is no clash and 
    # create files with no error. 
    p = PdfPages(filename+'.pdf')
      
    # get_fignums Return list of existing 
    # figure numbers 
    fig_nums = plt.get_fignums()   
    figs = [plt.figure(n) for n in fig_nums] 
      
    # iterating over the numbers in list 
    for fig in figs:  
        
        # and saving the files 
        fig.savefig(p, format='pdf', dpi=300)
        # fig.savefig(filename+'.png', format='png', dpi=300)
          
    # close the object 
    p.close() 



def plot_fig3(
        session_data,
        max_sessions=25
        ):
    

    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    outcomes = outcomes[start_idx:]
    dates = dates[start_idx:]
    session_id = np.arange(len(outcomes)) + 1
    filename = './figures/'+subject+'/'+'fig2_'+subject+'_avg_trajectory'    
    
    print()
    
    for i in range(session_data['total_sessions']):
        # indices = [ind for ind, ele in enumerate(session_data['outcomes'][0]) if ele == 'Reward']
        print(i)
        
        numRewardedTrials = len(session_data['rewarded_trials'][i])
        
        # plt.plot(session_data['encoder_time_aligned'], session_data['encoder_pos_avg'][i],'-')
        encoder_times_vis1 = session_data['encoder_times_aligned_VisStim1']
        encoder_pos_avg_vis1 = session_data['encoder_pos_avg_vis1'][i]
        encoder_times_vis2 = session_data['encoder_times_aligned_VisStim2']
        encoder_pos_avg_vis2 = session_data['encoder_pos_avg_vis2'][i]
        encoder_times_rew = session_data['encoder_times_aligned_Reward']
        encoder_pos_avg_rew = session_data['encoder_pos_avg_rew'][i]
        
        time_left_VisStim1 = session_data['time_left_VisStim1']
        time_right_VisStim1 = session_data['time_right_VisStim1']
        
        time_left_VisStim2 = session_data['time_left_VisStim2']
        time_right_VisStim2 = session_data['time_right_VisStim2']
        
        time_left_rew = session_data['time_left_rew']
        time_right_rew = session_data['time_right_rew']
        
        target_thresh = session_data['session_target_thresh']
        
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
        fig.subplots_adjust(hspace=0.7)
        fig.suptitle(subject + ' - ' + dates[i] + ' - Average Joystick Trajectories. '  + str(numRewardedTrials) + ' Trials Rewarded.' )
        
        # vis 1 aligned
        axs[0].plot(encoder_times_vis1, encoder_pos_avg_vis1,'-', label='Average Trajectory')
        axs[0].axvline(x = 0, color = 'r', label = 'VisStim1', linestyle='--')
        axs[0].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
        
        axs[0].set_title('VisStim1 Aligned.\n')
        
        axs[0].legend(loc='upper right')
        # axs[0].legend(bbox_to_anchor=(0.70, 1), loc=2, borderaxespad=0.)
                
        # axs[0].set_xlim(time_left_VisStim1, time_right_VisStim1)
        axs[0].set_xlim(time_left_VisStim1, 6.0)
        axs[0].set_ylim(-0.2, target_thresh+1)
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['top'].set_visible(False)
        axs[0].set_xlabel('trial time relative to VisStim1 [s]')
        axs[0].set_ylabel('joystick deflection [deg]')
        
        # plt.subplots_adjust(hspace=0.7)
        # plt.plot(encoder_times_vis1, encoder_pos_avg_vis1,'-', label='Average Trajectory')
        # plt.axvline(x = 0, color = 'r', label = 'VisStim1', linestyle='--')
        # plt.axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
        # axs1.set_title(subject + ' - ' + dates[i])
        # fig1.suptitle('VisStim1 Aligned. ' + str(numRewardedTrials) + ' Trials Rewarded. Average Joystick Trajectory.' )
        # fig1.tight_layout()
        # fig1.legend(loc='center right')
        # plt.xlim(time_left_VisStim1, time_right_VisStim1)
        # plt.ylim(-0.2, target_thresh+1)
        # axs1.spines['right'].set_visible(False)
        # axs1.spines['top'].set_visible(False)
        # axs1.set_xlabel('trial time relative to VisStim1 [s]')
        # axs1.set_ylabel('joystick deflection [deg]')
        # plt.show()
        
        
        # vis 2 aligned
        # fig2, axs2 = plt.subplots(1, figsize=(10, 4))
        # axs[1].subplots_adjust(hspace=0.7)
        axs[1].plot(encoder_times_vis2, encoder_pos_avg_vis2,'-', label='Average Trajectory')
        axs[1].axvline(x = 0, color = 'r', label = 'VisStim2', linestyle='--')
        axs[1].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
        # axs[1].set_title(subject + ' - ' + dates[i])
        axs[1].set_title('VisStim2 Aligned.\n')
        # axs[1].tight_layout()  
        axs[1].legend(loc='upper right')
        # plt.xlim(time_left_VisStim2, time_right_VisStim2)
        axs[1].set_xlim(-0.5, 4.0)
        axs[1].set_ylim(-0.2, target_thresh+1)
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
        axs[1].set_xlabel('trial time relative to VisStim2 [s]')
        axs[1].set_ylabel('joystick deflection [deg]')
        
        # fig2, axs2 = plt.subplots(1, figsize=(10, 4))
        # plt.subplots_adjust(hspace=0.7)
        # plt.plot(encoder_times_vis2, encoder_pos_avg_vis2,'-', label='Average Trajectory')
        # plt.axvline(x = 0, color = 'r', label = 'VisStim2', linestyle='--')
        # plt.axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
        # axs2.set_title(subject + ' - ' + dates[i])
        # fig2.suptitle('VisStim2 Aligned. ' + str(numRewardedTrials) + ' Trials Rewarded. Average Joystick Trajectory.' )
        # fig2.tight_layout()  
        # fig2.legend(loc='center right')
        # # plt.xlim(time_left_VisStim2, time_right_VisStim2)
        # plt.xlim(-0.5, 4.0)
        # plt.ylim(-0.2, target_thresh+1)
        # axs2.spines['right'].set_visible(False)
        # axs2.spines['top'].set_visible(False)
        # axs2.set_xlabel('trial time relative to VisStim1 [s]')
        # axs2.set_ylabel('joystick deflection [deg]')
        # plt.show()
        
        # reward aligned
        # fig3, axs3 = plt.subplots(1, figsize=(10, 4))
        # axs[2].subplots_adjust(hspace=0.7)
        axs[2].plot(encoder_times_rew, encoder_pos_avg_rew,'-', label='Average Trajectory')
        axs[2].axvline(x = 0, color = 'r', label = 'Reward', linestyle='--')
        axs[2].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
        # axs[2].set_title(subject + ' - ' + dates[i])
        axs[2].set_title('Reward Aligned.\n')
        # axs[2].tight_layout()    
        axs[2].legend(loc='upper right')       
        # plt.xlim(time_left_rew, time_right_rew)
        axs[2].set_xlim(-1.0, 4.0)
        axs[2].set_ylim(-0.2, target_thresh+1)
        axs[2].spines['right'].set_visible(False)
        axs[2].spines['top'].set_visible(False)
        axs[2].set_xlabel('trial time relative to Reward [s]')
        axs[2].set_ylabel('joystick deflection [deg]')
        
        
        fig.tight_layout()
        # fig3, axs3 = plt.subplots(1, figsize=(10, 4))
        # plt.subplots_adjust(hspace=0.7)
        # plt.plot(encoder_times_rew, encoder_pos_avg_rew,'-', label='Average Trajectory')
        # plt.axvline(x = 0, color = 'r', label = 'Reward', linestyle='--')
        # plt.axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
        # axs3.set_title(subject + ' - ' + dates[i])
        # fig3.suptitle('Reward Aligned. ' + str(numRewardedTrials) + ' Trials Rewarded. Average Joystick Trajectory.' )
        # fig3.tight_layout()    
        # fig3.legend(loc='center right')   
        # # plt.xlim(time_left_rew, time_right_rew)
        # plt.xlim(-1.0, time_right_rew)
        # plt.ylim(-0.2, target_thresh+1)
        # axs3.spines['right'].set_visible(False)
        # axs3.spines['top'].set_visible(False)
        # axs3.set_xlabel('trial time relative to VisStim1 [s]')
        # axs3.set_ylabel('joystick deflection [deg]')
        # plt.show()        
        print()
        os.makedirs('./figures/'+subject+'/'+dates[i], exist_ok = True)
        save_image(filename)
        fig.savefig('./figures/'+subject+'/'+dates[i]+'/fig2_'+subject+'_avg_trajectory_vis1.png', dpi=300)
        # fig2.savefig('./figures/'+subject+'/'+dates[i]+'/fig2_'+subject+'_avg_trajectory_vis2.png', dpi=300)
        # fig3.savefig('./figures/'+subject+'/'+dates[i]+'/fig2_'+subject+'_avg_trajectory_rew.png', dpi=300)
        # fig1.close()
        # fig2.close()
        # fig3.close()
        
    # axs.tick_params(tick1On=False)
    # axs.spines['left'].set_visible(False)
    # axs.spines['right'].set_visible(False)
    # axs.spines['top'].set_visible(False)
    # axs.yaxis.grid(True)
    # axs.set_xlabel('training session')
    # axs.set_ylabel('number of trials')
    # axs.set_xticks(np.arange(len(outcomes))+1)
    # axs.set_xticklabels(dates, rotation='vertical')
    # axs.set_title(subject)
    # axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    # fig.suptitle('average joystick trajectory for' + 'rewarded trials. VisStim1 aligned.' )
    # fig.tight_layout()
    
    
    print('Completed fig2 for ' + subject)
    print()
    # filename = './figures/fig2_'+subject+'_avg_trajectory'
    # save_image(filename)
    # fig.savefig('./figures/fig2_'+subject+'_avg_trajectory.pdf', dpi=300)
    # fig.savefig('./figures/fig2_'+subject+'_avg_trajectory.png', dpi=300)
    plt.close()
    
# session_data = session_data_1
# plot_fig2(session_data)