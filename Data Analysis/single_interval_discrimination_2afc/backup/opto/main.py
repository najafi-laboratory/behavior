import DataIO
from plot.fig1_psychometric import plot_fig1


if __name__ == "__main__":

    subject_list = ['VM1', 'VM5']

    session_data = DataIO.run(subject_list)

    for plotter in [
            plot_fig1,
            ]:
            plotter(session_data)
