import DataIO
from plot.fig1_outcome import plot_fig1
from plot.fig2_complete_trials_percentage import plot_fig2
from plot.fig3_percentage_epoch import plot_fig3
from plot.fig4_psychometric_epoch import plot_fig4
from plot.fig5_stim_isi import plot_fig5
from plot.fig6_av_sync import plot_fig6
from plot.fig7_early_iti import plot_fig7
from plot.p1_psychometric import plot_p1
from plot.p2_reaction import plot_p2


if __name__ == "__main__":

    session_data_1 = DataIO.read_trials('VM1')
    session_data_2 = DataIO.read_trials('VM4')
    session_data_3 = DataIO.read_trials('VM5')
    session_data_4 = DataIO.read_trials('VM6')

    for plotter in [
            plot_fig1,
            plot_fig2,
            plot_fig3,
            plot_fig4,
            plot_fig5,
            plot_fig6,
            plot_fig7,
            ]:
        plotter(
         	session_data_1,
         	session_data_2,
         	session_data_3,
         	session_data_4)
    '''
    plot_p1(session_data_1, session_data_2)
    plot_p2(session_data_1, session_data_2)
    '''
    
    
    
    
    
    