
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt

# importing csv module
import csv
# Import pandas
import pandas as pd
import numpy as np
  
# reading csv file 
# camlog_df = pd.read_csv("C:/data analysis/behavior/cam_scope_sync/Cam Scope Tests/Updated Labcams/2/VG01_V1_20240517_seq2b4o2_t_cam0_run001_20240517_151926.csv")
camlog_df = pd.read_csv("C:/data analysis/behavior/cam_scope_sync/Cam Scope Tests/Updated Labcams/2/VG01_V1_20240517_seq2b4o2_t_cam0_run001_20240517_151926.csv")
# df = pd.read_csv("C:/data analysis/behavior/cam_scope_sync/Cam Scope Tests/Orig Labcams/1/orig test 1_cam0_run000_20240608_151556.camlog")
print(camlog_df.head())
print(camlog_df.tail())
# df = pd.DataFrame(data=data).T
# s = df.to_csv()
# print(s)

# frame_id,timestamp,line
cam_frame_id_list = camlog_df['frame_id'].values.tolist()
cam_timestamp_list = camlog_df['timestamp'].values
cam_gpio_list = camlog_df['line'].values.tolist()
# cam_frame_id_list = camlog_df['frame_id'].values.tolist()
# cam_timestamp_list = camlog_df['timestamp'].values
# cam_gpio_list = camlog_df['line'].values.tolist()

cam_timestamp_marker_list = 5 * np.ones(len(cam_frame_id_list), dtype = int)

# zero and convert camlog from s to ms
# cam_timestamp_list = cam_timestamp_list.to_numpy()
cam_timestamp_list = (cam_timestamp_list - cam_timestamp_list[0]) * 1000

print("cam_frame_id_list:", cam_frame_id_list)
print("cam_timestamp_list:", cam_timestamp_list)
print("cam_gpio_list:", cam_gpio_list)

print("len cam_frame_id_list:", len(cam_frame_id_list))
print("len cam_timestamp_list:", len(cam_timestamp_list))
print("len cam_gpio_list:", len(cam_gpio_list))
 
# reading csv file 
vr_df = pd.read_csv("C:/data analysis/behavior/cam_scope_sync/Cam Scope Tests/Updated Labcams/2/VG01_V1_20240517_seqf2b4o2_t-497_Cycle00001_VoltageRecording_001.csv")
# df = pd.read_csv("C:/data analysis/behavior/cam_scope_sync/Cam Scope Tests/Orig Labcams/1/orig test 1_cam0_run000_20240608_151556.camlog")
print(vr_df.head())
print(vr_df.tail())
# df = pd.DataFrame(data=data).T
# s = df.to_csv()
# print(s)

# frame_id,timestamp,line
scope_timestamp_list = vr_df['Time(ms)'].values.tolist()
scope_start_list = vr_df[' Input 0'].values.tolist()
scope_pd_list = vr_df[' Input 1'].values
etl_list = vr_df[' Input 3'].values.tolist()
# scope_cam_list = vr_df[' Input 5'].values[0:30000]

# scope_pd_list =  (scope_pd_list / max(scope_pd_list)) * 5.10
# scope_cam_list =  (scope_cam_list / max(scope_cam_list)) * 4.98


##################################################
# read camlog
camlog_df = pd.read_csv("C:/data analysis/behavior/cam_scope_sync/Cam Scope Tests/Updated Labcams/2/FN13_20240613_js_t_cam0_run006_20240613_102257.csv")
cam_timestamp_list = camlog_df['timestamp'].values

# read 2p frames
vr_df = pd.read_csv("C:/data analysis/behavior/cam_scope_sync/Cam Scope Tests/Updated Labcams/2/FN13_20240613_js_DCNCNO_t-025_Cycle00001_VoltageRecording_001.csv")
etl_list = vr_df[' Input 3'].values



len(cam_timestamp_list), len(etl_list)

# get frame onsets on 2p voltage trace
above_threshold = np.array(etl_list) > 1

# Find the difference between consecutive elements in the boolean array
# The onset corresponds to a change from False to True
onset_indices = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1

len(onset_indices)
np.diff(onset_indices)


e = np.array(len(etl_list)-1)/2
time_2p_voltage_trace = np.arange(0, e, .5)


time_onset_indices = time_2p_voltage_trace[onset_indices]
time_onset_indices 
np.diff(time_onset_indices)


t = time_onset_indices - time_onset_indices[0]
c = cam_timestamp_list - cam_timestamp_list[0]
c = c*1000

l = min(len(t), len(c))
d = c[:l] - t[:l]
np.sort(d)
min(d), max(d)

##################################################




print("scope_timestamp_list:", scope_timestamp_list[0:4])
print("scope_start_list:", scope_start_list[0:4])
print("scope_pd_list:", scope_pd_list[0:4])
print("etl_list:", etl_list[0:4])
# print("scope_cam_list:", scope_cam_list[0:4])

print("len scope_timestamp_list:", len(scope_timestamp_list))
print("len scope_start_list:", len(scope_start_list))
print("len scope_pd_list:", len(scope_pd_list))
print("len etl_list:", len(etl_list))
# print("len scope_cam_list:", len(scope_cam_list))


plt.figure(dpi=2200)

plt.plot(scope_timestamp_list, etl_list, linewidth=0.05)
plt.plot(scope_timestamp_list, scope_start_list, linewidth=0.1)
plt.plot(scope_timestamp_list, scope_pd_list, linewidth=0.01)
# plt.plot(scope_timestamp_list, scope_cam_list, linewidth=0.1)
# plt.plot(cam_timestamp_list, cam_frame_id_list, 'o')
plt.plot(cam_timestamp_list, cam_timestamp_marker_list, ',', markersize=10)

# Saving the figure.
# plt.savefig("VG_output.png")

trigger_val = 3.3
scope_timestamp_onset = np.flatnonzero((np.array(etl_list[:-1]) < trigger_val) & (np.array(etl_list[1:]) > trigger_val))+1

plt.show()




#t_camlog_interp


# csv file name
# filename = "C:/data analysis/behavior/cam_scope_sync/Cam Scope Tests/Orig Labcams/1/orig_test_1_cam0_run000_20240608_151556.csv"
 
# initializing the titles and rows list
# fields = []
# rows = []

# file_object = open(filename, "r")
def find_falling_edges(etl_list, threshold=1):
    # Create a boolean array where True indicates the voltage is above the threshold
    above_threshold = etl_list > threshold

    # Find the difference between consecutive elements in the boolean array
    # Falling edge corresponds to a change from True to False
    falling_edges_indices = np.where(np.diff(above_threshold.astype(int)) == -1)[0] + 1

    return falling_edges_indices

# Example usage
# etl_list = np.array([0.1, 0.3, 0.2, 1.4, 1.6, 1.4, 1.7, 0, 0.3])
falling_edges_indices = find_falling_edges(etl_list)

print("Falling edges indices:", falling_edges_indices)

"""
# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
 
    # extracting field names through first row
    fields = next(csvreader)
 
    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)
 
    # get total number of rows
    print("Total no. of rows: %d" % (csvreader.line_num))
 
# printing the field names
print('Field names are:' + ', '.join(field for field in fields))
 
# printing first 5 rows
print('/nFirst 5 rows are:/n')
for row in rows[:5]:
    # parsing each column of a row
    for col in row:
        print("%10s" % col, end=" "),
    print('/n')
"""