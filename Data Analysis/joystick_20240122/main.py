import DataIO
import tkinter as tk
from tkinter import filedialog


from plot.fig1_outcome import plot_fig1
from plot.fig2_trajectory_avg_sess import plot_fig2
from plot.fig3_trajectories import plot_fig3
from plot.fig4_trajectory_avg_sess_superimpose import plot_fig4
# from plot.fig2_complete_trials_percentage import plot_fig2
# from plot.fig3_percentage_epoch import plot_fig3
# from plot.fig4_psychometric_epoch import plot_fig4
# from plot.fig5_stim_isi import plot_fig5
# from plot.fig6_av_sync import plot_fig6
# from plot.fig7_early_iti import plot_fig7
# from plot.fig8_reaction_trial import plot_fig8
# from plot.fig9_reaction_sess import plot_fig9
# from plot.fig10_com import plot_fig10

# session_data_1 = session_data_1
# session_data_2 = session_data_2
# session_data_3 = session_data_3
# session_data_4 = session_data_4

if __name__ == "__main__":

    
    extract_data = 1
    run_plots = 0
    
    if extract_data:
        window = tk.Tk()
        window.wm_attributes('-topmost', 1)
        window.withdraw()   # this supress the tk window
        
        session_data_path = 'C:\\behavior\\session_data'
        file_paths_1 = list(filedialog.askopenfilenames(parent=window, initialdir=session_data_path+'\\YH4', title='Select YH4 Sessions'))
        file_paths_2 = list(filedialog.askopenfilenames(parent=window, initialdir=session_data_path+'\\YH5', title='Select YH5 Sessions'))
        file_paths_3 = list(filedialog.askopenfilenames(parent=window, initialdir=session_data_path+'\\FN10', title='Select FN10 Sessions'))
        file_paths_4 = list(filedialog.askopenfilenames(parent=window, initialdir=session_data_path+'\\FN13', title='Select FN13 Sessions'))
        
        session_data_1 = DataIO.read_trials('YH4', file_paths_1)
        session_data_2 = DataIO.read_trials('YH5', file_paths_2)
        session_data_3 = DataIO.read_trials('FN10', file_paths_3)
        session_data_4 = DataIO.read_trials('FN13', file_paths_4)

    if run_plots:
        for plotter in [
                plot_fig1,
                plot_fig2,
                plot_fig3,
                plot_fig4,
                # plot_fig5,
                # plot_fig6,
                # plot_fig7,
                # plot_fig8,
                # plot_fig9,
                # plot_fig10,
                ]:
            for session_data in [
                    session_data_1,
                    session_data_2,
                    session_data_3,
                    session_data_4,
                    # session_data_5,
                    # session_data_6
                    ]:
                plotter(session_data)
 
    
 