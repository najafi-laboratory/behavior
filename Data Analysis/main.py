import DataIO
from plot.fig1_complete_trials_percentage import plot_fig1
from plot.fig2_outcome import plot_fig2
from plot.fig3_percentage_epoch import plot_fig3
from plot.fig4_psychometric_epoch import plot_fig4
from plot.fig5_stim_isi import plot_fig5
from plot.fig6_av_sync import plot_fig6


if __name__ == "__main__":

    session_data_1 = DataIO.read_trials('VM1')
    session_data_2 = DataIO.read_trials('VM4')
    session_data_3 = DataIO.read_trials('VM5')
    session_data_4 = DataIO.read_trials('VM6')

    plot_fig1(
     	session_data_1,
     	session_data_2,
     	session_data_3,
     	session_data_4)

    plot_fig2(
     	session_data_1,
     	session_data_2,
     	session_data_3,
     	session_data_4)

    plot_fig3(
     	session_data_1,
     	session_data_2,
     	session_data_3,
     	session_data_4)

    plot_fig4(
     	session_data_1,
     	session_data_2,
     	session_data_3,
     	session_data_4)

    plot_fig5(
     	session_data_1,
     	session_data_2,
     	session_data_3,
     	session_data_4)

    plot_fig6(
     	session_data_1,
     	session_data_2,
     	session_data_3,
     	session_data_4)

