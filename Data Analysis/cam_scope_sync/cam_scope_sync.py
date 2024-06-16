
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt

# importing csv module
import csv
# Import pandas
import pandas as pd
import numpy as np
  
# reading csv file 
camlog_df = pd.read_csv("C:/data analysis/behavior/cam_scope_sync/Cam Scope Tests/Orig Labcams/1/orig_test_1_cam0_run000_20240608_151556.csv")
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

cam_timestamp_marker_list = np.ones(len(cam_frame_id_list), dtype = int)

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
vr_df = pd.read_csv("C:/data analysis/behavior/cam_scope_sync/Cam Scope Tests/Orig Labcams/1/orig test 1-601/orig test 1-601_Cycle00001_VoltageRecording_001.csv")
# df = pd.read_csv("C:/data analysis/behavior/cam_scope_sync/Cam Scope Tests/Orig Labcams/1/orig test 1_cam0_run000_20240608_151556.camlog")
print(vr_df.head())
print(vr_df.tail())
# df = pd.DataFrame(data=data).T
# s = df.to_csv()
# print(s)

# frame_id,timestamp,line
scope_timestamp_list = vr_df['Time(ms)'].values.tolist()
etl_list = vr_df[' Input 3'].values.tolist()


print("scope_timestamp_list:", scope_timestamp_list[0:4])
print("etl_list:", etl_list[0:4])

print("len scope_timestamp_list:", len(scope_timestamp_list))
print("len etl_list:", len(etl_list))


plt.plot(scope_timestamp_list, etl_list)
plt.plot(cam_timestamp_list, cam_frame_id_list, 'o')
plt.plot(cam_timestamp_list, cam_timestamp_marker_list, 'o')
plt.show()


# csv file name
# filename = "C:/data analysis/behavior/cam_scope_sync/Cam Scope Tests/Orig Labcams/1/orig_test_1_cam0_run000_20240608_151556.csv"
 
# initializing the titles and rows list
# fields = []
# rows = []

# file_object = open(filename, "r")


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