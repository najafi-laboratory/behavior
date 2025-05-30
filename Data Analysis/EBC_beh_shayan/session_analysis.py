import os
import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec
import h5py
from scipy.ndimage import uniform_filter1d
from scipy.spatial import ConvexHull

from utils.alignment import fec_cr_aligned, moving_average, sort_numbers_as_strings, fec_zero, fec_crop
from utils.indication import find_max_with_gradient, CR_stat_indication, CR_FEC, block_type, find_index, cr_onset_calc
from utils.save_plots import save_fec_plots_to_pdf


beh_folder = "./data/beh"
early_trials_check = "early"
static_threshold = 0.03
mice = [folder for folder in os.listdir(beh_folder) if folder != ".DS_Store"] 
mice_cr_performance_file = './outputs/beh/mice_cr_performance.pdf'
x_grid = 4
y_grid = 7
fig = plt.figure(figsize=(x_grid * 7, y_grid * 7))
gs = GridSpec(y_grid, x_grid)

for mouse_i, mouse in enumerate(mice):
    # all_sessions = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(beh_folder, mouse,"processed"))if file != ".DS_Store"]
    session_path = os.path.join(beh_folder, mouse, "processed")
    all_sessions = sorted([
        os.path.splitext(file)[0]
        for file in os.listdir(session_path)
        if not file.startswith('.')
    ])
    i = 0
    if not os.path.exists(f'./outputs/beh/{mouse}'):
            os.mkdir(f'./outputs/beh/{mouse}')
            print(f"data folder '{mouse}' created.")
    else:
        print(f"data folder '{mouse}' already exists.")
    
    slopes = []
    sig_id_all = {}
    # Lists to hold percentages across sessions
    no_cr_percentages = []
    cr_percentages = []
    poor_cr_percentages = []
    session_labels = []

    for i, session_date in enumerate(all_sessions):
        print(f'working on the session {session_date}')
        folder_path = f'./outputs/beh/evaluation/{mouse}/{session_date}'
        os.makedirs(folder_path, exist_ok=True)

        individual_fec_file = os.path.join(folder_path, f'{session_date}_indi_fec.pdf')

        trials = h5py.File(f"./data/beh/{mouse}/processed/{session_date}.h5")["trial_id"]
        fec, fec_time, trials = fec_zero(trials)
        shorts, longs = block_type(trials)
        CR_stat, CR_interval_avg, base_line_avg, cr_interval_idx, bl_interval_idx  = CR_stat_indication(trials, fec, fec_time , static_threshold, AP_delay=10)
        all_id = sort_numbers_as_strings(shorts + longs)
        # save_fec_plots_to_pdf(trials, fec_time, fec, CR_stat, all_id, filename=individual_fec_file)

        # Count CR types
        no_cr_count = sum(1 for id in all_id if CR_stat[id] == 0)
        cr_count = sum(1 for id in all_id if CR_stat[id] == 1)
        poor_cr_count = sum(1 for id in all_id if CR_stat[id] == 2)
        
        total = no_cr_count + cr_count + poor_cr_count
        if total == 0:
            continue  # Skip empty sessions

        # Store percentages
        no_cr_percentages.append((no_cr_count / total) * 100)
        cr_percentages.append((cr_count / total) * 100)
        poor_cr_percentages.append((poor_cr_count / total) * 100)
        session_labels.append(session_date)

        print(f'session {session_date} added')
        # break

        # if i > 20:
        #     break

    # Convert to numpy arrays
    no_cr = np.array(no_cr_percentages)
    cr = np.array(cr_percentages)
    poor_cr = np.array(poor_cr_percentages)

    
    x = np.arange(len(session_labels))
    ax_0 = fig.add_subplot(gs[2 * mouse_i: 2 * mouse_i + 1, 0:4])

    ax_0.bar(x, cr, label='CR', color='red')
    ax_0.bar(x, poor_cr, bottom=cr, label='Poor CR', color='purple')
    ax_0.bar(x, no_cr, bottom=cr + poor_cr, label='No CR', color='blue')
    ax_0.spines['top'].set_visible(False)
    ax_0.spines['right'].set_visible(False)

    # Annotate each bar
    for i in range(len(x)):
        total = cr[i] + poor_cr[i] + no_cr[i]

    ax_0.set_xticks(x)
    ax_0.set_xticklabels(session_labels, rotation=75, ha = 'right', fontsize=5)
    ax_0.set_ylim(0, 100)
    ax_0.set_ylabel('Percentage of CR status (%)')
    ax_0.set_title(f'{mouse}')
    ax_0.legend(loc='upper right', fontsize=5)
# plt.tight_layout()
with PdfPages(mice_cr_performance_file) as pdf:
    pdf.savefig(fig)
    plt.close(fig)

        
        # if i > 5:
             # break

        # break

        # short_fec_0, short_fec_1, short_fec_2 = [], [], []
        
        # # plot the averages to check if things are looking correct.
        # for id in shorts:
        #     if CR_stat[id] == 0:
        #         short_fec_0.append(fec[id])
        #     elif CR_stat[id] == 1:
        #         short_fec_1.append(fec[id])
        #     elif CR_stat[id] == 2:
        #         short_fec_2.append(fec[id])

        # short_fec_0_avg = np.mean(short_fec_0, axis=0)
        # short_fec_1_avg = np.mean(short_fec_1, axis=0)
        # short_fec_2_avg = np.mean(short_fec_2, axis=0)
