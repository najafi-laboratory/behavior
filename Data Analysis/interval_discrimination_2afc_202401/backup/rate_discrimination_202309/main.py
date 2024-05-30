import DataIO
from plot.fig1_outcome import plot_fig1
from plot.fig2_complete_trials_percentage import plot_fig2
from plot.fig4_psychometric_epoch import plot_fig4
from plot.fig8_reaction_trial import plot_fig8
from plot.fig9_reaction_sess import plot_fig9


if __name__ == "__main__":

    subject_list = [
        'YH7', 'YH10']

    session_data = DataIO.run(subject_list)

    for plotter in [
            plot_fig1,
            plot_fig2,
            plot_fig4,
            plot_fig8,
            plot_fig9,
            ]:
            plotter(session_data)
