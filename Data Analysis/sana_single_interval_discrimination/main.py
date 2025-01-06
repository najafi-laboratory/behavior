#!/usr/bin/env python3

import os
import sys
import fitz
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from datetime import datetime
import random
from scipy.stats import sem
import psytrack as psy
import pickle
import DataIO
import DataIO_all
import DataIOPsyTrack


from plot import plot_outcome
from plot import plot_complete_trials
from plot import plot_psychometric_post
# from plot import plot_psychometric_post_no_naive
from plot import plot_psychometric_pre
from plot import plot_psychometric_epoch
from plot import plot_psychometric_percep
from plot import plot_reaction_time
# from plot import plot_reaction_time_no_naive
from plot import plot_decision_time
from plot import plot_decision_time_side
from plot import plot_decision_time_sessions
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
from plot import plot_side_outcome_percentage
from plot import plot_psytrack_bias
from plot import plot_psytrack_performance


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
    # Get the current date
    current_date = datetime.now()
    # Format the date as 'yyyymmdd'
    formatted_date = current_date.strftime('%Y%m%d')
    
    
    session_data_path = 'C:\\behavior\\session_data'
    # session_data_path = 'C:\\localscratch\\behavior\\session_data'
    # output_dir_onedrive = './figures/'
    output_dir_onedrive = 'C:\\Users\\timst\\OneDrive - Georgia Institute of Technology\\Najafi_Lab\\2__Data_Analysis\\Behavior\\Single_Interval_Discrimination\\'
    # output_dir_local = './figures/'
    output_dir_local = 'C:\\Users\\timst\\OneDrive - Georgia Institute of Technology\\Desktop\\PHD\\SingleIntervalDiscrimination\\FIGS\\'
    # last_day = '20241215'
    #subject_list = ['YH7', 'YH10', 'LG03', 'VT01', 'FN14' , 'LG04' , 'VT02' , 'VT03']
    # subject_list = ['LCHR_TS01', 'LCHR_TS02']
    subject_list = ['LCHR_TS01', 'LCHR_TS02', 'LG08_TS03', 'LG09_TS04', 'LG11_TS05']
    # subject_list = ['LCHR_TS01_update']
    # subject_list = ['LCHR_TS02_update']
    # subject_list = ['LCHR_TS01_update', 'LCHR_TS02_update']
    # subject_list = ['LCHR_TS02']
    # subject_list = ['LG09_TS04']
    # subject_list = ['LG09_TS04_update']

    M = DataIOPsyTrack.run(subject_list , session_data_path)

    save_file = 0
    load_file = 0
    
    if save_file:
        M = DataIOPsyTrack.run(subject_list , session_data_path)
        #######
        #  save extracted/processed data
        ######
        # Save the variable to a file
        with open('M.pkl', 'wb') as file:
            for item in M:
                pickle.dump(item, file)
    
    if load_file:
        with open('M.pkl', 'rb') as file:
            M = pickle.load(file)
         
    

    subject_report = fitz.open()
    subject_session_data = M[0]
    # subject_session_data = M
    
    # Chemo
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
        else:
            Chemo[dates.index('2024' + ch)] = 0
    
    M[0]['Chemo'] = Chemo
    
    
    # start date of non-naive
    NonNaive = {'LCHR_TS01_update': '20241222',
                'LCHR_TS01': '20241203',
                'LCHR_TS02': '20241203',
                'LCHR_TS02_update': '20241203',
                'LG08_TS03': '20241228',
                'LG09_TS04': '20241227',
                'LG09_TS04_update': '20241222',
                'LG11_TS05': '20241230'}
    
    # Start date for averaging
    StartDate = {'LCHR_TS01_update': '20241222',
                'LCHR_TS01': '20241222',
                'LCHR_TS02': '20241222',
                'LCHR_TS02_update': '20241222',
                'LG08_TS03': '20241228',
                'LG09_TS04': '20241227',
                'LG09_TS04_update': '20241225',
                'LG11_TS05': '20241216'}

    # MoveCorrectSpout - First Session 
    MoveCorrectSpoutStart = {'LCHR_TS01_update': '20241222',
                             'LCHR_TS01': '20241214',
                             'LCHR_TS02': '20241214',
                             'LCHR_TS02_update': '20241214',
                             'LG08_TS03': '20241215',
                             'LG09_TS04': '20241221',
                             'LG09_TS04_update': '20241225',
                             'LG11_TS05': '20241218'}
    
    # Start date for averaging
    StartDate = {'LCHR_TS01_update': '20241226',
                'LCHR_TS01': '20241226',
                'LCHR_TS02': '20241226',
                'LCHR_TS02_update': '20241226',
                'LG08_TS03': '20241226',
                'LG09_TS04': '20241226',
                'LG09_TS04_update': '20241230',
                'LG11_TS05': '20241226'}    
    
    # add start dates to session data
    for i in range(len(M)):
        M[i]['non_naive'] = NonNaive[M[i]['name']]
        M[i]['start_date'] = StartDate[M[i]['name']]
        M[i]['move_correct_spout'] = MoveCorrectSpoutStart[M[i]['name']]
    

    
 
    ##########
    # PsyTrack
    size = len(M)
    weights = [{}] * size  # List of empty dictionaries
    K = [[]] * size  # List of zeros 
    hyper = [{}] * size  # List of empty dictionaries
    optList = [''] * size  # List of empty strings
    new_M = [{}] * size  # List of empty dictionaries
    hyp = [{}] * size  # List of empty dictionaries
    evd = [[]] * size  # List of zeros 
    wMode = [[]] * size  # List of zeros 
    hess_info = [{}] * size  # List of empty dictionaries
    
    for i in range(len(M)):
        # MoveCorrectSpout - First Session           
        # MCSS_idx = M[i]['dates'].index(MoveCorrectSpoutStart[M[i]['name']])
        # M[i]['move_correct_spout'] = MoveCorrectSpoutStart[M[i]['name']]
        
        M[i]['inputs'] = {}

        weights[i] = {'bias': 1}  # a special key
                    # 's1': 1,    # use only the first column of s1 from inputs
                    # 's2': 1}    # use only the first column of s2 from inputs
                    
        # weights.append({'bias': 1})  # a special key
        #             # 's1': 1,    # use only the first column of s1 from inputs
        #             # 's2': 1}    # use only the first column of s2 from inputs                    

        
        # It is often useful to have the total number of weights K in your model
        weights_at_i = weights[i]
        # K = np.sum([weights[i] for i in weights.keys()])
        K[i] = np.sum([weights_at_i[j] for j in weights_at_i.keys()])
        # K.append(np.sum([weights_at_i[j] for j in weights_at_i.keys()]))

        hyper[i] = {'sigInit': 2**4.,      # Set to a single, large value for all weights. Will not be optimized further.
                'sigma': [2**-4.]*K[i],   # Each weight will have it's own sigma optimized, but all are initialized the same
                'sigDay': M[i]['dayLength']}        # Indicates that session boundaries will be ignored in the optimization
        # hyper.append({'sigInit': 2**4.,      # Set to a single, large value for all weights. Will not be optimized further.
        #         'sigma': [2**-4.]*K[i],   # Each weight will have it's own sigma optimized, but all are initialized the same
        #         'sigDay': M[i]['dayLength']})        # Indicates that session boundaries will be ignored in the optimization
            

        optList[i] = ['sigma']
        # optList[i] = ['sigma']

        # new_M[i] = psy.trim(M[i], END=100000)  # trim dataset to first 10,000 trials

        # hyp[i], evd[i], wMode[i], hess_info[i] = psy.hyperOpt(new_M[i], hyper[i], weights[i], optList[i])
        
        
    
    for i in range(len(M)):
        M[i]['Chemo'] = np.zeros(M[i]['total_sessions'])  
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 6, figure=fig)

        plot_complete_trials.run(plt.subplot(gs[0, 0:3]), M[i])
        plot_side_outcome_percentage.run(plt.subplot(gs[1, 0:3]), M[i])
        plot_right_left_percentage.run(plt.subplot(gs[2, 0:3]), M[i])        

        plot_psychometric_epoch.run([plt.subplot(gs[j, 3]) for j in range(3)], M[i])
       
        plot_psychometric_post.run(plt.subplot(gs[0, 4]), M[i], start_from='std')
        plot_psychometric_post.run(plt.subplot(gs[1, 4]), M[i], start_from='start_date')
        plot_psychometric_post.run(plt.subplot(gs[2, 4]), M[i], start_from='non_naive')
       
        plot_decision_time.run(plt.subplot(gs[0, 5]), M[i], start_from='std')                  
        plot_decision_time.run(plt.subplot(gs[1, 5]), M[i], start_from='start_date')
        plot_decision_time.run(plt.subplot(gs[2, 5]), M[i], start_from='non_naive')        

        plot_psytrack_bias.run(plt.subplot(gs[3, 0:3]), M[i])  
        plot_psytrack_performance.run(plt.subplot(gs[3, 3:6]), M[i])        
              
        
        plt.suptitle(M[i]['subject'])
        fname = os.path.join(str(i).zfill(4)+'.pdf')
        fig.set_size_inches(30, 15)
        fig.savefig(fname, dpi=300)
        plt.close()
        roi_fig = fitz.open(fname)
        subject_report.insert_pdf(roi_fig)
        roi_fig.close()
        os.remove(fname)

                    
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 6, figure=fig)
        
        
        plot_decision_time_side.run(plt.subplot(gs[0, 1]), M[i], start_from='std')                  
        plot_decision_time_isi.run(plt.subplot(gs[0, 2]), M[i], start_from='start_date')
        # plot_decision_time_sessions.run(plt.subplot(gs[1:2, 0:6]), M[i], max_rt=700, plot_type='std', start_from='std')
        # plot_decision_time_sessions.run(plt.subplot(gs[2:3, 0:6]), M[i], max_rt=700, plot_type='lick-side', start_from='std')
        plot_decision_time_sessions.run(plt.subplot(gs[1:2, 0:6]), M[i], max_rt=700, plot_type='std', start_from='start_date')
        plot_decision_time_sessions.run(plt.subplot(gs[2:3, 0:6]), M[i], max_rt=700, plot_type='lick-side', start_from='start_date')
                
        
        # plot_decision_time_sessions.run(plt.subplot(gs[3:4, 0:6]), M[i], plot_type='choice-side', start_from='std')
        
        # plot_reaction_time.run(plt.subplot(gs[0, 5]), M[i], start_from='std')                  
        # plot_reaction_time.run(plt.subplot(gs[1, 5]), M[i], start_from='start_date')
        # plot_reaction_time.run(plt.subplot(gs[2, 5]), M[i], start_from='non_naive')        
        
        
        # plot_outcome.run(plt.subplot(gs[0, 0:3]), M[i])        
        # plot_early_lick_outcome.run(plt.subplot(gs[3, 3:5]), M[i])          
        
        # plot_psychometric_post_no_naive.run(plt.subplot(gs[1, 5]), M[i])
        # plot_psychometric_percep.run(plt.subplot(gs[3, 2]), M[i])
        
        # plot_reaction_time_no_naive.run(plt.subplot(gs[0, 5]), M[i])
        # plot_reaction_outcome.run(plt.subplot(gs[0, 5]), M[i])
        
        # plot_reaction_time.run(plt.subplot(gs[1, 4]), M[i])
        # plot_decision_time.run(plt.subplot(gs[1, 4]), M[i])
        # plot_decision_outcome.run(plt.subplot(gs[1, 5]), M[i])
                    #plot_strategy.run(plt.subplot(gs[2, 5]), M[i])
        # plot_decision_time_isi.run(plt.subplot(gs[2, 4]), M[i])
        # plot_reaction_time_isi.run(plt.subplot(gs[2, 5]), M[i])
                    # plot_short_long_percentage.run(plt.subplot(gs[2, 0:2]), M[i])

        
        # fig_bias = psy.plot_bias(new_M[i])

        # Access all axes in the figure using fig.get_axes()
        # axes = fig_bias.get_axes()
        # Modify the properties of the first subplot (axs[0])
        # axes[0].set_yticklabels(['Left', 0, 'Right'])
         

        plt.suptitle(M[i]['subject'])
        fname = os.path.join(str(i).zfill(4)+'.pdf')
        fig.set_size_inches(30, 15)
        fig.savefig(fname, dpi=300)
        plt.close()
        roi_fig = fitz.open(fname)
        subject_report.insert_pdf(roi_fig)
        roi_fig.close()
        os.remove(fname)
    # subject_report.save(output_dir_onedrive+subject_list[0]+'\\'+subject_list[0]+'_'+last_day+'_result_clean.pdf')
    subject_report.save(output_dir_onedrive+'single_interval_report'+'_'+formatted_date+'.pdf')
    subject_report.save(output_dir_local+'single_interval_report'+'_'+formatted_date+'.pdf')
    subject_report.close()
    for i in range(len(M)):
        plot_trial_outcomes.run(M[i],output_dir_onedrive, output_dir_local,formatted_date)
        #plot_category_each_session.run(M[i],output_dir_onedrive, output_dir_local,last_day)
        
        
        # import os
        # import sys
        # os.makedirs(output_figs_dir, exist_ok = True)
        # os.makedirs(output_imgs_dir, exist_ok = True)




#%%

if 0:

    session_data_path = 'C:\\behavior\\session_data\\SampleRat\\'
    # subject_list = ['SampleRat']
    # path = session_data_path + '\\' + subject_list
    
    # Extract premade dataset from npz
    D = np.load(session_data_path + 'sampleRatData.npz', allow_pickle=True)['D'].item()
    
    print("The keys of the dict for this example animal:\n   ", list(D.keys()))
    
    print("The shape of y:   ", D['y'].shape)
    print("The number of trials:   N =", D['y'].shape[0])
    print("The unique entries of y:   ", np.unique(D['y']))
    
    weights = {'bias': 1,  # a special key
                's1': 1,    # use only the first column of s1 from inputs
                's2': 1}    # use only the first column of s2 from inputs
    
    # It is often useful to have the total number of weights K in your model
    K = np.sum([weights[i] for i in weights.keys()])
    
    
    hyper= {'sigInit': 2**4.,      # Set to a single, large value for all weights. Will not be optimized further.
            'sigma': [2**-4.]*K,   # Each weight will have it's own sigma optimized, but all are initialized the same
            'sigDay': None}        # Indicates that session boundaries will be ignored in the optimization
    
    optList = ['sigma']
    
    new_D = psy.trim(D, END=10000)  # trim dataset to first 10,000 trials
    
    hyp, evd, wMode, hess_info = psy.hyperOpt(new_D, hyper, weights, optList)
    
    fig = psy.plot_weights(wMode, weights)
    
    fig = psy.plot_weights(wMode, weights, days=new_D["dayLength"], errorbar=hess_info["W_std"])
    
    fig_perf = psy.plot_performance(new_D)
    fig_bias = psy.plot_bias(new_D)



#%%
import DataIOPsyTrack

session_data_path = 'C:\\behavior\\session_data\\SampleRat\\'
# subject_list = ['SampleRat']
# path = session_data_path + '\\' + subject_list

# Extract premade dataset from npz
# D = np.load(session_data_path + 'sampleRatData.npz', allow_pickle=True)['D'].item()

session_data_path = 'C:\\behavior\\session_data'
subject_list = ['LCHR_TS01_update']

# M = DataIOPsyTrack.run(subject_list , session_data_path)

M[0]['inputs'] = {}

weights = {'bias': 1}  # a special key
            # 's1': 1,    # use only the first column of s1 from inputs
            # 's2': 1}    # use only the first column of s2 from inputs

# It is often useful to have the total number of weights K in your model
K = np.sum([weights[i] for i in weights.keys()])

hyper= {'sigInit': 2**4.,      # Set to a single, large value for all weights. Will not be optimized further.
        'sigma': [2**-4.]*K,   # Each weight will have it's own sigma optimized, but all are initialized the same
        'sigDay': M[0]['dayLength']}        # Indicates that session boundaries will be ignored in the optimization

optList = ['sigma']

new_M[0] = psy.trim(M[0], END=10000)  # trim dataset to first 10,000 trials

hyp, evd, wMode, hess_info = psy.hyperOpt(new_M[0], hyper, weights, optList)

#%%

fig = psy.plot_weights(wMode, weights)

fig = psy.plot_weights(wMode, weights, days=new_M[0]["dayLength"], errorbar=hess_info["W_std"])

fig_perf = psy.plot_performance(new_M[0])
# Access all axes in the figure using fig.get_axes()
axes = fig_perf.get_axes()
# Modify the properties of the first subplot (axs[0])
axes[0].set_ylabel("Choice Accuracy")



fig_bias = psy.plot_bias(new_M[0])

# Access all axes in the figure using fig.get_axes()
axes = fig_bias.get_axes()
# Modify the properties of the first subplot (axs[0])
axes[0].set_yticklabels(['Left', 0, 'Right'])

# axes.yticks([-0.5, 0.5], ['Left', 'Right'])


# Modify the properties of the first subplot (axs[0])
# axes[0].set_title("Top Plot")
# axes[0].set_ylabel("Y-axis for top plot")

# Modify the properties of the second subplot (axs[1])
# axes[1].set_title("Bottom Plot")
# axes[1].set_ylabel("Y-axis for bottom plot")

#%%
import warnings
warnings.filterwarnings('ignore')
if __name__ == "__main__":
    # Get the current date
    current_date = datetime.now()
    # Format the date as 'yyyymmdd'
    formatted_date = current_date.strftime('%Y%m%d')
    
    
    session_data_path = 'C:\\behavior\\session_data'
    # session_data_path = 'C:\\localscratch\\behavior\\session_data'
    # output_dir_onedrive = './figures/'
    output_dir_onedrive = 'C:\\Users\\timst\\OneDrive - Georgia Institute of Technology\\Najafi_Lab\\0_Data_analysis\\Behavior\\Single_Interval_Discrimination\\'
    # output_dir_local = './figures/'
    output_dir_local = 'C:\\Users\\timst\\OneDrive - Georgia Institute of Technology\\Desktop\\PHD\\SingleIntervalDiscrimination\\FIGS\\'
    # last_day = '20241215'
    #subject_list = ['YH7', 'YH10', 'LG03', 'VT01', 'FN14' , 'LG04' , 'VT02' , 'VT03']
    # subject_list = ['LCHR_TS01', 'LCHR_TS02']
    subject_list = ['LCHR_TS01', 'LCHR_TS02', 'LG08_TS03', 'LG11_TS05', 'LG09_TS04']

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
        else:
            Chemo[dates.index('2024' + ch)] = 0
    
    session_data[0]['Chemo'] = Chemo
    ##########
    
    for i in range(len(session_data)):
        session_data[i]['Chemo'] = np.zeros(session_data[i]['total_sessions'])  
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 6, figure=fig)
        # plot_outcome.run(plt.subplot(gs[0, 0:3]), session_data[i])
        plot_complete_trials.run(plt.subplot(gs[0, 0:3]), session_data[i])
        # plot_early_lick_outcome.run(plt.subplot(gs[3, 3:5]), session_data[i])
        plot_psychometric_post.run(plt.subplot(gs[2, 2]), session_data[i])
        # plot_psychometric_percep.run(plt.subplot(gs[3, 2]), session_data[i])
        plot_psychometric_epoch.run([plt.subplot(gs[j, 3]) for j in range(3)], session_data[i])
        plot_reaction_time.run(plt.subplot(gs[0, 4]), session_data[i])
        # plot_reaction_outcome.run(plt.subplot(gs[0, 5]), session_data[i])
        
        # plot_reaction_time.run(plt.subplot(gs[1, 4]), session_data[i])
        # plot_decision_time.run(plt.subplot(gs[1, 4]), session_data[i])
        # plot_decision_outcome.run(plt.subplot(gs[1, 5]), session_data[i])
                    #plot_strategy.run(plt.subplot(gs[2, 5]), session_data[i])
        # plot_decision_time_isi.run(plt.subplot(gs[2, 4]), session_data[i])
        # plot_reaction_time_isi.run(plt.subplot(gs[2, 5]), session_data[i])
                    # plot_short_long_percentage.run(plt.subplot(gs[2, 0:2]), session_data[i])
        plot_side_outcome_percentage.run(plt.subplot(gs[1, 0:3]), session_data[i])
        plot_right_left_percentage.run(plt.subplot(gs[2, 0:2]), session_data[i])
        
        
        
        plt.suptitle(session_data[i]['subject'])
        fname = os.path.join(str(i).zfill(4)+'.pdf')
        fig.set_size_inches(30, 15)
        fig.savefig(fname, dpi=300)
        plt.close()
        roi_fig = fitz.open(fname)
        subject_report.insert_pdf(roi_fig)
        roi_fig.close()
        os.remove(fname)
    # subject_report.save(output_dir_onedrive+subject_list[0]+'\\'+subject_list[0]+'_'+last_day+'_result_clean.pdf')
    subject_report.save(output_dir_onedrive+'single_interval_report'+'_'+formatted_date+'.pdf')
    subject_report.save(output_dir_local+'single_interval_report'+'_'+formatted_date+'.pdf')
    subject_report.close()
    for i in range(len(session_data)):
        plot_trial_outcomes.run(session_data[i],output_dir_onedrive, output_dir_local,formatted_date)
        #plot_category_each_session.run(session_data[i],output_dir_onedrive, output_dir_local,last_day)
        
        
        # import os
        # import sys
        # os.makedirs(output_figs_dir, exist_ok = True)
        # os.makedirs(output_imgs_dir, exist_ok = True)



#%%
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 140

import psytrack as psy

seed = 31
num_weights = 4
num_trials = 5000
hyper = {'sigma'   : 2**np.array([-4.0,-5.0,-6.0,-7.0]),
         'sigInit' : 2**np.array([ 0.0, 0.0, 0.0, 0.0])}

# Simulate
simData = psy.generateSim(K=num_weights, N=num_trials, hyper=hyper,
                          boundary=6.0, iterations=1, seed=seed, savePath=None)

# Plot
psy.plot_weights(simData['W'].T);
plt.ylim(-3.6,3.6);

rec = psy.recoverSim(simData)

psy.plot_weights(rec['wMode'], errorbar=rec["hess_info"]["W_std"])
plt.plot(simData['W'], c="k", ls="-", alpha=0.5, lw=0.75, zorder=0)
plt.ylim(-3.6,3.6);

true_sigma = np.log2(rec['input']['sigma'])
avg_sigma = np.log2(rec['hyp']['sigma'])
err_sigma = rec['hess_info']['hyp_std']

plt.figure(figsize=(2,2))
colors = np.unique(list(psy.COLORS.values()))
for i in range(num_weights):
    plt.plot(i, true_sigma[i], color="black", marker="_", markersize=12, zorder=0)
    plt.errorbar([i], avg_sigma[i], yerr=2*err_sigma[i], color=colors[i], lw=1, marker='o', markersize=5)

plt.xticks([0,1,2,3]); plt.yticks(np.arange(-8,-2))
plt.gca().set_xticklabels([r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$", r"$\sigma_4$"])
plt.xlim(-0.5,3.5); plt.ylim(-7.5,-3.5)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.ylabel(r"$\log_2(\sigma)$");



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

