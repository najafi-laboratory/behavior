#!/usr/bin/env python3

import os
import fitz
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

import DataIO
import DataIO_all
from plot import plot_outcome
from plot import plot_complete_trials
from plot import plot_psychometric_post
from plot import plot_psychometric_pre
from plot import plot_psychometric_epoch
from plot import plot_psychometric_percep
from plot import plot_reaction_time
from plot import plot_decision_time
from plot import plot_reaction_outcome
from plot import plot_decision_outcome
from plot import plot_single_trial_licking
from plot import plot_psychometric_post_early_included
from plot import plot_strategy
from plot import plot_decision_time_isi
from plot import plot_reaction_time_isi
from plot import plot_trial_outcomes
from plot import plot_short_long_percentage
from plot import plot_right_left_percentage
from plot import plot_category_each_session
from plot import plot_average_licking
from plot import plot_early_lick_outcome

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import os
import fitz
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

import DataIO
import DataIO_all
from plot_strategy import count_isi_flash
from plot_strategy import count_psychometric_curve
from plot_strategy import count_short_long
from plot_strategy import strategy
from plot_strategy import count_isi_decision_time
from plot_strategy import count_flash_decision_time
from plot_strategy import strategy_epoch
from plot_strategy import decision_time_dist

#%%
import warnings
warnings.filterwarnings('ignore')
if __name__ == "__main__":
    session_data_path = 'C:\\behavior\\session_data'
    output_dir_onedrive = './figures/'
    output_dir_local = './figures/'
    last_day = '20241210'
    #subject_list = ['YH7', 'YH10', 'LG03', 'VT01', 'FN14' , 'LG04' , 'VT02' , 'VT03']
    subject_list = ['LCHR_TS01', 'LCHR_TS02']

    session_data = DataIO.run(subject_list , session_data_path)
    

    subject_report = fitz.open()
    subject_session_data = session_data[0]
    # subject_session_data = session_data
    
    #########
    Chemo = np.zeros(subject_session_data['total_sessions'])
    if subject_list[0] == 'YH7':
        chemo_sessions = ['0627' , '0701' , '0625' , '0623' , '0613' , '0611' , '0704' , '0712' , '0716' , '0725' , '0806' , '0809' , '0814' , '0828' , '0821' ,] #YH7
    elif subject_list[0] == 'VT01':
        chemo_sessions = ['0618' , '0701' , '0704' , '0710' , '0712' , '0715' , '0718'] #VT01
    elif subject_list[0] == 'LG03':
        chemo_sessions = ['0701' , '0704' , '0709' , '0712' , '0726', '0731', '0806' , '0809' , '0814' , '0828' , '0821'] #LG03
    elif subject_list[0] == 'LG04':
        chemo_sessions = ['0712' , '0717' , '0725', '0731', '0806' , '0814' , '0828' , '0821'] #LG04
    else:
        chemo_sessions = []
    dates = subject_session_data['dates']
    for ch in chemo_sessions:
        if '2024' + ch in dates:
            Chemo[dates.index('2024' + ch)] = 1
    session_data[0]['Chemo'] = Chemo
    ##########
    
    for i in range(len(session_data)):
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 6, figure=fig)
        # plot_outcome.run(plt.subplot(gs[0, 0:3]), session_data[i])
        # plot_complete_trials.run(plt.subplot(gs[1, 0:3]), session_data[i])
        # plot_early_lick_outcome.run(plt.subplot(gs[3, 3:5]), session_data[i])
        # plot_psychometric_post.run(plt.subplot(gs[2, 2]), session_data[i])
        # plot_psychometric_percep.run(plt.subplot(gs[3, 2]), session_data[i])
        # plot_psychometric_epoch.run([plt.subplot(gs[j, 3]) for j in range(3)], session_data[i])
        # plot_reaction_time.run(plt.subplot(gs[0, 4]), session_data[i])
        # plot_reaction_outcome.run(plt.subplot(gs[0, 5]), session_data[i])
        # plot_decision_time.run(plt.subplot(gs[1, 4]), session_data[i])
        # plot_decision_outcome.run(plt.subplot(gs[1, 5]), session_data[i])
        #             #plot_strategy.run(plt.subplot(gs[2, 5]), session_data[i])
        # plot_decision_time_isi.run(plt.subplot(gs[2, 4]), session_data[i])
        # plot_reaction_time_isi.run(plt.subplot(gs[2, 5]), session_data[i])
                    # plot_short_long_percentage.run(plt.subplot(gs[2, 0:2]), session_data[i])
        plot_right_left_percentage.run(plt.subplot(gs[3, 0:2]), session_data[i])
        plt.suptitle(session_data[i]['subject'])
        fname = os.path.join(str(i).zfill(4)+'.pdf')
        fig.set_size_inches(30, 15)
        fig.savefig(fname, dpi=300)
        plt.close()
        roi_fig = fitz.open(fname)
        subject_report.insert_pdf(roi_fig)
        roi_fig.close()
        os.remove(fname)
    subject_report.save(output_dir_onedrive+subject_list[0]+'\\'+subject_list[0]+'_'+last_day+'_result_clean.pdf')
    subject_report.close()
    for i in range(len(session_data)):
        plot_trial_outcomes.run(session_data[i],output_dir_onedrive, output_dir_local,last_day)
        #plot_category_each_session.run(session_data[i],output_dir_onedrive, output_dir_local,last_day)

#%%

import warnings
warnings.filterwarnings('ignore')
if __name__ == "__main__":

    subject_report = fitz.open()

    
    for i in range(len(session_data)):
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 6, figure=fig)
        
        strategy.run(plt.subplot(gs[0, 0]), session_data[i])
        count_short_long.run(plt.subplot(gs[2, 0]), plt.subplot(gs[1, 0]), session_data[i])
        count_psychometric_curve.run(plt.subplot(gs[0, 1]), plt.subplot(gs[1, 1]),plt.subplot(gs[2, 1]),plt.subplot(gs[3, 1]),session_data[i])
        count_isi_flash.run(plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2]),plt.subplot(gs[2, 2]),plt.subplot(gs[3, 2]), session_data[i])
        strategy_epoch.run([plt.subplot(gs[j, 3]) for j in range(4)], session_data[i])
        plt.suptitle(session_data[i]['subject']+' count strategy analysis')
        fname = os.path.join(str(i).zfill(4)+'.pdf')
        fig.set_size_inches(30, 15)
        fig.savefig(fname, dpi=300)
        plt.close()
        roi_fig = fitz.open(fname)
        subject_report.insert_pdf(roi_fig)
        roi_fig.close()
        os.remove(fname)
    subject_report.save(output_dir_onedrive+subject_list[0]+'\\'+subject_list[0]+'_'+last_day+'_count_strategy_clean.pdf')
    subject_report.close()
    
    
#%%

import warnings
warnings.filterwarnings('ignore')
if __name__ == "__main__":

    subject_report = fitz.open()

    
    for i in range(len(session_data)):
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 6, figure=fig)
        plot_decision_time.run(plt.subplot(gs[0, 0]), session_data[i])
        count_isi_decision_time.run(plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2]),plt.subplot(gs[2, 2]),plt.subplot(gs[3, 2]), session_data[i])
        count_flash_decision_time.run(plt.subplot(gs[0, 1]), plt.subplot(gs[1, 1]),plt.subplot(gs[2, 1]),plt.subplot(gs[3, 1]), session_data[i])
        decision_time_dist.run([plt.subplot(gs[j, 3]) for j in range(4)], session_data[i])
        
        plt.suptitle(session_data[i]['subject']+' desicion time analysis')
        fname = os.path.join(str(i).zfill(4)+'.pdf')
        fig.set_size_inches(30, 15)
        fig.savefig(fname, dpi=300)
        plt.close()
        roi_fig = fitz.open(fname)
        subject_report.insert_pdf(roi_fig)
        roi_fig.close()
        os.remove(fname)
    subject_report.save(output_dir_onedrive+subject_list[0]+'\\'+subject_list[0]+'_'+last_day+'_count_decision_time_clean.pdf')
    subject_report.close()

#%%

import warnings
warnings.filterwarnings('ignore')
if __name__ == "__main__":
    session_data_path = 'C:\\Users\\Sana\\OneDrive\\Desktop\\PHD\\IntervalDiscrimination\\Data\\20240805\\'
    output_dir_onedrive = 'C:\\Users\\Sana\\OneDrive\\Desktop\\PHD\\IntervalDiscrimination\\FIGs\\20240805\\'
    output_dir_local = 'C:\\Users\\Sana\\OneDrive\\Desktop\\PHD\\IntervalDiscrimination\\FIGs\\20240805\\'
    last_day = '20240805'
    #subject_list = ['YH7', 'YH10', 'LG03', 'VT01', 'FN14' , 'LG04' , 'VT02' , 'VT03']
    subject_list = ['VT03']

    session_data = DataIO_all.run(subject_list , session_data_path)
    

    subject_report = fitz.open()
    subject_session_data = session_data[0]
    
    #########
    Chemo = np.zeros(subject_session_data['total_sessions'])
    if subject_list[0] == 'YH7':
        chemo_sessions = ['0627' , '0701' , '0625' , '0623' , '0613' , '0611' , '0704' , '0712' , '0716' , '0725', '0806' , '0809' , '0814'] #YH7
    elif subject_list[0] == 'VT01':
        chemo_sessions = ['0618' , '0701' , '0704' , '0710' , '0712' , '0715' , '0718'] #VT01
    elif subject_list[0] == 'LG03':
        chemo_sessions = ['0701' , '0704' , '0709' , '0712' , '0726', '0806' , '0809' , '0814'] #LG03
    elif subject_list[0] == 'LG04':
        chemo_sessions = ['0712' , '0717' , '0725', '0806' , '0814'] #LG04
    else:
        chemo_sessions = []
    dates = subject_session_data['dates']
    for ch in chemo_sessions:
        if '2024' + ch in dates:
            Chemo[dates.index('2024' + ch)] = 1
    session_data[0]['Chemo'] = Chemo
    ##########
    
    for i in range(len(session_data)):
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 6, figure=fig)
        plot_outcome.run(plt.subplot(gs[0, 0:3]), session_data[i])
        plot_complete_trials.run(plt.subplot(gs[1, 0:3]), session_data[i])
        plot_early_lick_outcome.run(plt.subplot(gs[3, 3:5]), session_data[i])
        plot_psychometric_post.run(plt.subplot(gs[2, 2]), session_data[i])
        plot_psychometric_percep.run(plt.subplot(gs[3, 2]), session_data[i])
        plot_psychometric_epoch.run([plt.subplot(gs[j, 3]) for j in range(3)], session_data[i])
        plot_reaction_time.run(plt.subplot(gs[0, 4]), session_data[i])
        plot_reaction_outcome.run(plt.subplot(gs[0, 5]), session_data[i])
        plot_decision_time.run(plt.subplot(gs[1, 4]), session_data[i])
        plot_decision_outcome.run(plt.subplot(gs[1, 5]), session_data[i])
        #plot_strategy.run(plt.subplot(gs[2, 5]), session_data[i])
        plot_decision_time_isi.run(plt.subplot(gs[2, 4]), session_data[i])
        plot_reaction_time_isi.run(plt.subplot(gs[2, 5]), session_data[i])
        plot_short_long_percentage.run(plt.subplot(gs[2, 0:2]), session_data[i])
        plot_right_left_percentage.run(plt.subplot(gs[3, 0:2]), session_data[i])
        plt.suptitle(session_data[i]['subject'])
        fname = os.path.join(str(i).zfill(4)+'.pdf')
        fig.set_size_inches(30, 15)
        fig.savefig(fname, dpi=300)
        plt.close()
        roi_fig = fitz.open(fname)
        subject_report.insert_pdf(roi_fig)
        roi_fig.close()
        os.remove(fname)
    subject_report.save(output_dir_onedrive+subject_list[0]+'\\'+subject_list[0]+'_'+last_day+'_result_all.pdf')
    subject_report.close()

#%%

import warnings
warnings.filterwarnings('ignore')
if __name__ == "__main__":

    subject_report = fitz.open()

    
    for i in range(len(session_data)):
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 6, figure=fig)
        
        strategy.run(plt.subplot(gs[0, 0]), session_data[i])
        count_short_long.run(plt.subplot(gs[2, 0]), plt.subplot(gs[1, 0]), session_data[i])
        count_psychometric_curve.run(plt.subplot(gs[0, 1]), plt.subplot(gs[1, 1]),plt.subplot(gs[2, 1]),plt.subplot(gs[3, 1]),session_data[i])
        count_isi_flash.run(plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2]),plt.subplot(gs[2, 2]),plt.subplot(gs[3, 2]), session_data[i])
        count_isi_decision_time.run(plt.subplot(gs[0, 3]), plt.subplot(gs[1, 3]),plt.subplot(gs[2, 3]),plt.subplot(gs[3, 3]), session_data[i])
        count_flash_decision_time.run1(plt.subplot(gs[0, 4]), plt.subplot(gs[1, 4]),plt.subplot(gs[2, 4]),plt.subplot(gs[3, 4]), session_data[i])
        strategy_epoch.run([plt.subplot(gs[j, 5]) for j in range(4)], session_data[i])
        
        plt.suptitle(session_data[i]['subject']+' count strategy analysis')
        fname = os.path.join(str(i).zfill(4)+'.pdf')
        fig.set_size_inches(30, 15)
        fig.savefig(fname, dpi=300)
        plt.close()
        roi_fig = fitz.open(fname)
        subject_report.insert_pdf(roi_fig)
        roi_fig.close()
        os.remove(fname)
    subject_report.save(output_dir_onedrive+subject_list[0]+'\\'+subject_list[0]+'_'+last_day+'_count_strategy_all.pdf')
    subject_report.close()
    
#%%

import warnings
warnings.filterwarnings('ignore')
import os
import fitz
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import DataIO
if __name__ == "__main__":
    
    session_data_path = 'C:\\Users\\Sana\\OneDrive\\Desktop\\PHD\\IntervalDiscrimination\\Data\\20240819\\'
    subject_list = ['YH7', 'YH10', 'LG03', 'VT01', 'FN14' , 'LG04' , 'VT02' , 'VT03']
    subject_list = ['LG04']

    session_data = DataIO.run(subject_list , session_data_path)
    output_dir_onedrive = 'C:\\Users\\Sana\\OneDrive\\Desktop\\PHD\\IntervalDiscrimination\\FIGs\\20240819\\'
    output_dir_local = 'C:\\Users\\Sana\\OneDrive\\Desktop\\PHD\\IntervalDiscrimination\\FIGs\\20240819\\'

    for i in range(len(session_data)):
        subject_session_data = session_data[0]
        plot_single_trial_licking.run(subject_session_data,output_dir_onedrive, output_dir_local)
        plot_average_licking.run(subject_session_data,output_dir_onedrive, output_dir_local)    

#%%


if __name__ == "__main__":

    subject_list = ['LCHR_TS01', 'LCHR_TS02', 'LG08_TS03']

    session_data = DataIO.run(subject_list)
    subject_session_data = session_data[0]

    subject_report = fitz.open()
    for i in range(len(session_data)):
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(3, 6, figure=fig)
        plot_outcome.run(plt.subplot(gs[0, 0:3]), session_data[i])
        plot_complete_trials.run(plt.subplot(gs[1, 0:3]), session_data[i])
        plot_psychometric_post.run(plt.subplot(gs[2, 0]), session_data[i])
        plot_psychometric_percep.run(plt.subplot(gs[2, 1]), session_data[i])
        plot_psychometric_pre.run(plt.subplot(gs[2, 2]), session_data[i])
        plot_psychometric_epoch.run([plt.subplot(gs[i, 3]) for i in range(3)], session_data[i])
        plot_reaction_time.run(plt.subplot(gs[0, 4]), session_data[i])
        plot_reaction_outcome.run(plt.subplot(gs[0, 5]), session_data[i])
        plot_decision_time.run(plt.subplot(gs[1, 4]), session_data[i])
        plot_decision_outcome.run(plt.subplot(gs[1, 5]), session_data[i])
        plt.suptitle(session_data[i]['subject'])
        fname = os.path.join(str(i).zfill(4)+'.pdf')
        fig.set_size_inches(30, 15)
        fig.savefig(fname, dpi=300)
        plt.close()
        roi_fig = fitz.open(fname)
        subject_report.insert_pdf(roi_fig)
        roi_fig.close()
        os.remove(fname)
     
    # Get the current date
    current_date = datetime.now()
    # Format the date as 'yyyymmdd'
    formatted_date = current_date.strftime('%Y%m%d')
    # Random integer between 1 and 100
    random_integer = str(random.randint(1000, 9999))  
    subject_report.save('./figures/single_interval_subject_report_'+formatted_date+'_'+random_integer+'.pdf')
    subject_report.close()

