from fig1_outcome import plot_fig1
from fig2_complete_trials_percentage import plot_fig2
from fig3_percentage_epoch import plot_fig3
from fig4_psychometric_epoch import plot_fig4
from fig5_stim_isi import plot_fig5
from fig6_av_sync import plot_fig6
from fig7_com import plot_fig7
from fig8_reaction_trial import plot_fig8
from fig9_reaction_sess import plot_fig9


def run(session_data):
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
        for data in session_data:
            plotter(data)