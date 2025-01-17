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
from plot import plot_psychometric_post
from plot import plot_psychometric_pre
from plot import plot_psychometric_epoch
from plot import plot_psychometric_percep
from plot import plot_reaction_time
from plot import plot_decision_time
from plot import plot_reaction_outcome
from plot import plot_decision_outcome

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

