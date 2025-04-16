import random
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py

from utils.alignment import moving_average, sort_numbers_as_strings, fec_zero
from utils.indication import CR_stat_indication, block_type, cr_onset_calc, find_max_with_gradient

beh_folder = "./data/beh"

for mouse in os.listdir(beh_folder):
    all_sessions = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(beh_folder, mouse, "processed"))]
    output_folder = f'./outputs/beh/{mouse}/validation'
    os.makedirs(output_folder, exist_ok=True)
    
    with PdfPages(os.path.join(output_folder, f'{mouse}CR_indication_validation.pdf')) as pdf:
        for session_date in all_sessions:
            if session_date.startswith('.'):
                continue
            print(session_date)
            
            trials = h5py.File(f"./data/beh/{mouse}/processed/{session_date}.h5")["trial_id"]
            fec, fec_time_0, trials = fec_zero(trials)
            fec = moving_average(fec, window_size=10)
            shorts, longs = block_type(trials)
            
            all_id = sort_numbers_as_strings(shorts + longs)
            if len(all_id) < 3:
                sample_ids = all_id  # If less than 3 trials, use all available
            else:
                sample_ids = random.sample(all_id, 5)  # Randomly select 3 trials
            
            plt.figure(figsize=(35, 7))
            for i, trial_id in enumerate(sample_ids):
                airpuff = trials[trial_id]["AirPuff"][0] - trials[trial_id]["LED"][0]
                CR_stat, CR_interval_avg, base_line_avg, cr_interval_idx, bl_interval_idx = CR_stat_indication(trials, fec, fec_time_0, static_threshold=0.025, AP_delay=3)
                
                veolcity_threshold_fraction = 0.75
                amp_threshold_fraction = 0.05
                print("cr:", CR_stat[trial_id])
                cr_idx = cr_onset_calc(
                    fec[trial_id], fec_time_0[trial_id], 5, airpuff, CR_stat[trial_id])

                peak_time, peak_value, peak_index, gradients = find_max_with_gradient(fec_time_0[trial_id][bl_interval_idx[trial_id][1]: cr_interval_idx[trial_id][1]], fec[trial_id][bl_interval_idx[trial_id][1]: cr_interval_idx[trial_id][1]])
                
                plt.subplot(1, 5, i + 1)
                plt.plot(fec_time_0[trial_id], fec[trial_id], color='red' if CR_stat[trial_id] == 1 else 'blue')
                plt.xlim(-100, 500)
                plt.axvline(fec_time_0[trial_id][cr_idx], color='r', linestyle='--', label = 'CR onset')

                if peak_time:
                    plt.axvline(peak_time, color='g', linestyle='--', label = 'Peak time')

                plt.axvspan(0, trials[trial_id]["LED"][1] - trials[trial_id]["LED"][0], color="gray", alpha=0.5)
                plt.axvspan(trials[trial_id]["AirPuff"][0] - trials[trial_id]["LED"][0], 
                            trials[trial_id]["AirPuff"][1] - trials[trial_id]["LED"][0], 
                            color="lime" if trials[trial_id]['trial_type'][()] == 2 else "blue", alpha=0.5)
                plt.title(f"{session_date}\nTrial:{trial_id}")
                plt.xlabel("Time (ms)")
                plt.ylabel("FEC")
                plt.legend()
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    
    print(f"Plots saved in {output_folder}/{mouse}_session_plots.pdf")

