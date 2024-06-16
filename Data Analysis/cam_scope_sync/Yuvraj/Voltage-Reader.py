import numpy as np
import h5py
import pandas as pd
import cv2

video = "VG01_V1_20240517_seq2b4o2_t_cam0_run001_20240517_151926.avi"

def read_raw_voltages():
    f = h5py.File(
        'fn13_raw_voltages.h5',
        'r')
    vol_time = np.array(f['raw']['vol_time'])
    vol_start_bin = np.array(f['raw']['vol_start_bin'])
    vol_stim_bin = np.array(f['raw']['vol_stim_bin'])
    vol_img_bin = np.array(f['raw']['vol_img_bin'])
    f.close()
    return [vol_time, vol_start_bin, vol_stim_bin, vol_img_bin]


voltage = read_raw_voltages()
# print(voltage)
voltage_df = pd.DataFrame(columns=["time", "vol_img_bin"])
voltage_df["time"] = voltage[0]
voltage_df["vol_img_bin"] = voltage[3]

camlog = pd.read_excel("fn13_camlog.xlsx")

cap = cv2.VideoCapture(video)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_duration_ms = 1000 / fps

ms_per_frame_voltage = []
ms_per_frame_camlog = []
ms_per_frame_cv2 = []
ms_per_frame = pd.DataFrame(columns=["voltage", "camlog", "cv2"])

for index, row in voltage_df.iterrows():
    if index == 0:
        continue
    if row["vol_img_bin"] != 1.0 and len(ms_per_frame_voltage) == 0:
        ms_per_frame_voltage.append(row["time"])
    if row["vol_img_bin"] != 1.0 and row["time"] - ms_per_frame_voltage[-1] > 2:
        ms_per_frame_voltage.append(row["time"])
    if index == len(camlog):
        break

for index, row in camlog.iterrows():
    if index == 0:
        continue
    time = row["Time"] - camlog.iloc[0, 0]
    ms_per_frame_camlog.append(time * 1000)
    ms_per_frame_cv2.append(frame_duration_ms * index)

    # Needed to keep dataframe rows aligned (camlog collects data for 4 more rows which are ignored this way)
    if index == len(ms_per_frame_voltage):
        break

ms_per_frame["voltage"] = ms_per_frame_voltage
ms_per_frame["camlog"] = ms_per_frame_camlog
ms_per_frame["cv2"] = ms_per_frame_cv2

ms_per_frame.to_excel("fn13_voltage_camlog_frames_aligned.xlsx")