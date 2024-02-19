import DataIO
from plot.fig1_outcome import plot_fig1
from plot.fig2_complete_trials_percentage import plot_fig2
from plot.fig3_percentage_epoch import plot_fig3
from plot.fig4_psychometric_epoch import plot_fig4
from plot.fig5_stim_isi import plot_fig5
from plot.fig6_av_sync import plot_fig6
from plot.fig7_com import plot_fig7
from plot.fig8_reaction_trial import plot_fig8
from plot.fig9_reaction_sess import plot_fig9


if __name__ == "__main__":
    
    subject_list = [
        'FN11', 'FN12', 'FN14',
        'VM5',
        'YH6', 'YH7', 'YH8', 'YH9', 'YH10', 'YH11']

    session_data = DataIO.run(subject_list)
    
    for plotter in [
            plot_fig1,
            plot_fig2,
            plot_fig3,
            plot_fig4,
            plot_fig5,
            plot_fig6,
            plot_fig7,
            plot_fig8,
            plot_fig9,
            ]:
            plotter(session_data)
