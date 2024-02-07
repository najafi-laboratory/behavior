import DataIO
from plot import plotter


if __name__ == "__main__":
    
    subject_list = [
        'FN11', 'FN14',
        'VM5',
        'YH6', 'YH7', 'YH8', 'YH9', 'YH10', 'YH11']

    session_data = DataIO.run(subject_list)
    
    plotter.run(session_data)
