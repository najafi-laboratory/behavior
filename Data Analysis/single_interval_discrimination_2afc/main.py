#!/usr/bin/env python3

import os
import fitz
from datetime import datetime
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import DataIO
from plot import plot_outcome
from plot import plot_complete_trials
from plot import plot_side_outcome_percentage
from plot import plot_psychometric_post_emp
from plot import plot_psychometric_post_perc
from plot import plot_psychometric_pre_emp
from plot import plot_psychometric_pre_perc
from plot import plot_psychometric_epoch_emp
from plot import plot_psychometric_epoch_perc
from plot import plot_reaction_time
from plot import plot_decision_time
from plot import plot_decision_outcome

if __name__ == "__main__":

    pdf_plot = 1   
    pdf_plot2 = 1

    # subject_list = ['LCHR_TS02']
    # subject_list = ['LCHR_TS01', 'LCHR_TS02', 'LG08_TS03']
    subject_list = ['LCHR_TS01', 'LCHR_TS02']
    # subject_list = ['LCHR_TS02', 'YH10', 'LG03', 'VT01']

    session_data = DataIO.run(subject_list)
    subject_session_data = session_data[0]

    subject_report = fitz.open()
    for i in range(len(session_data)):
        fig = plt.figure(layout='constrained', figsize=(35, 20))
        gs = GridSpec(4, 7, figure=fig)
        # plot_outcome.run(plt.subplot(gs[0, 0:3]), session_data[i])
        # plot_complete_trials.run(plt.subplot(gs[1, 0:3]), session_data[i])
        plot_side_outcome_percentage.run(plt.subplot(gs[0, 0:3]), session_data[i])
        plot_psychometric_post_emp.run(plt.subplot(gs[1, 0]), session_data[i])
        plot_reaction_time.run(plt.subplot(gs[1, 1]), session_data[i])
        
        # plot_psychometric_epoch_emp.run([plt.subplot(gs[i, 3]) for i in range(3)], session_data[i])
        # plot_psychometric_post_emp.run(plt.subplot(gs[2, 0]), session_data[i])
    #     plot_psychometric_post_perc.run(plt.subplot(gs[3, 0]), session_data[i])
    #     plot_psychometric_pre_emp.run(plt.subplot(gs[2, 1]), session_data[i])
    #     plot_psychometric_pre_perc.run(plt.subplot(gs[3, 1]), session_data[i])
        # plot_psychometric_epoch_emp.run([plt.subplot(gs[i, 3]) for i in range(3)], session_data[i])
    #     plot_psychometric_epoch_perc.run([plt.subplot(gs[i, 4]) for i in range(3)], session_data[i])
        # plot_reaction_time.run(plt.subplot(gs[0, 5]), session_data[i])
        # plot_decision_time.run(plt.subplot(gs[1, 5]), session_data[i])
        # plot_decision_outcome.run(plt.subplot(gs[1, 6]), session_data[i])
        plt.suptitle(session_data[i]['subject'])
        if pdf_plot:
            fname = os.path.join(str(i).zfill(4)+'.pdf')
            fig.set_size_inches(35, 20)
            fig.savefig(fname, dpi=300)
            plt.close()
            roi_fig = fitz.open(fname)
            subject_report.insert_pdf(roi_fig)
            roi_fig.close()
            os.remove(fname)
    if pdf_plot:    
        # Get the current date
        current_date = datetime.now()
        # Format the date as 'yyyymmdd'
        formatted_date = current_date.strftime('%Y%m%d')
        # Random integer between 1 and 100
        random_integer = str(random.randint(1000, 9999))  
        subject_report.save('./figures/single_interval_subject_report_'+formatted_date+'_'+random_integer+'.pdf')
        subject_report.close()

    # subject_list = ['LCHR_TS02']
    # subject_list = ['LCHR_TS01', 'LCHR_TS02', 'LG08_TS03']
    subject_list = ['LCHR_TS01_daily', 'LCHR_TS02_daily', 'LG08_TS03_daily']
    # subject_list = ['LCHR_TS02', 'YH10', 'LG03', 'VT01']

    session_data = DataIO.run(subject_list)
    subject_session_data = session_data[0]
    subject_report = fitz.open()

    for i in range(len(session_data)):

        fig = plt.figure(layout='constrained', figsize=(35, 20))
        gs = GridSpec(4, 7, figure=fig)
        # plot_outcome.run(plt.subplot(gs[0, 0:3]), session_data[i])
        # plot_complete_trials.run(plt.subplot(gs[1, 0:3]), session_data[i])
        plot_side_outcome_percentage.run(plt.subplot(gs[0, 0:3]), session_data[i])
        plot_psychometric_post_emp.run(plt.subplot(gs[1, 0]), session_data[i])
        plot_reaction_time.run(plt.subplot(gs[1, 1]), session_data[i])
        
        # plot_psychometric_epoch_emp.run([plt.subplot(gs[i, 3]) for i in range(3)], session_data[i])
        # plot_psychometric_post_emp.run(plt.subplot(gs[2, 0]), session_data[i])
    #     plot_psychometric_post_perc.run(plt.subplot(gs[3, 0]), session_data[i])
    #     plot_psychometric_pre_emp.run(plt.subplot(gs[2, 1]), session_data[i])
    #     plot_psychometric_pre_perc.run(plt.subplot(gs[3, 1]), session_data[i])
        # plot_psychometric_epoch_emp.run([plt.subplot(gs[i, 3]) for i in range(3)], session_data[i])
    #     plot_psychometric_epoch_perc.run([plt.subplot(gs[i, 4]) for i in range(3)], session_data[i])
        # plot_reaction_time.run(plt.subplot(gs[0, 5]), session_data[i])
        # plot_decision_time.run(plt.subplot(gs[1, 5]), session_data[i])
        # plot_decision_outcome.run(plt.subplot(gs[1, 6]), session_data[i])
        plt.suptitle(session_data[i]['subject'])
        if pdf_plot2:
            fname = os.path.join(str(i).zfill(4)+'.pdf')
            fig.set_size_inches(35, 20)
            fig.savefig(fname, dpi=300)
            plt.close()
            roi_fig = fitz.open(fname)
            subject_report.insert_pdf(roi_fig)
            roi_fig.close()
            os.remove(fname)
    if pdf_plot2:    
        # Get the current date
        current_date = datetime.now()
        # Format the date as 'yyyymmdd'
        formatted_date = current_date.strftime('%Y%m%d')
        # Random integer between 1 and 100
        random_integer = str(random.randint(1000, 9999))  
        subject_report.save('./figures/single_interval_single_day_subject_report_'+formatted_date+'_'+random_integer+'.pdf')
        subject_report.close()