#!/usr/bin/env python3

import os
import tifffile
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime


# find the target filenames in ops['data_path'].
def list_filenames(
        ops
        ):
    # ch1: SESSION_Cycle00001_Ch1_000NUM.ome.tif.
    # ch2: SESSION_Cycle00001_Ch2_000NUM.ome.tif.
    # voltage: SESSION_Cycle00001_VoltageRecording_000NUM.csv.
    ch1_files = [f for f in os.listdir(ops['data_path'])
                 if 'Ch1' in f and '.tif' in f]
    ch2_files = [f for f in os.listdir(ops['data_path'])
                 if 'Ch2' in f and '.tif' in f]
    vol_record = [f for f in os.listdir(ops['data_path'])
                  if 'VoltageRecording' in f and '.csv' in f]
    # sort the filename based on the recording partition number.
    ch1_files.sort()
    ch2_files.sort()
    vol_record.sort()
    return ch1_files, ch2_files, vol_record


# read the tif files given the filename list.
def read_tif_to_np(
        ops,
        ch_files
        ):
    # read into list.
    ch_data = []
    for f in tqdm(ch_files):
        data = tifffile.imread(os.path.join(ops['data_path'], f))
        ch_data.append(data)
    # concat on time axis.
    if len(ch_data) > 0:
        ch_data = np.concatenate(ch_data, axis=0)
        ch_data = ch_data.astype('float32')
    # no channel data found.
    else:
        ch_data = np.zeros((1, ops['Lx'], ops['Ly']))
    return ch_data


# read the voltage recording file.
def read_vol_to_np(
        ops,
        vol_record
        ):
    df_vol = pd.read_csv(
        os.path.join(ops['data_path'],vol_record[0]),
        engine='python')
    # column 0: time index in ms.
    # column 1: trial start signal from bpod.
    # column 2: stimulus signal from photodiode.
    # column 3: BNC2 not in use.
    # column 4: image trigger signal from 2p scope camera.
    time      = df_vol.iloc[:,0].to_numpy()
    vol_start = df_vol.iloc[:,1].to_numpy()
    vol_stim  = df_vol.iloc[:,2].to_numpy()
    vol_img   = df_vol.iloc[:,4].to_numpy()
    return time, vol_start, vol_stim, vol_img


# main function for reading the data.
def run(ops):
    print('===============================================')
    print('========== read and merge video data ==========')
    print('===============================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ch1_files, ch2_files, vol_record = list_filenames(ops)
    print('Found {} files for channel 1.'.format(len(ch1_files)))
    print('Found {} files for channel 2.'.format(len(ch2_files)))
    print('Reading channel 1 data.')
    print(ch1_files)
    ch1_data = read_tif_to_np(ops, ch1_files)
    print('Reading channel 2 data.')
    print(ch2_files)
    ch2_data = read_tif_to_np(ops, ch2_files)
    print('Reading voltage recordings.')
    time, vol_start, vol_stim, vol_img = read_vol_to_np(ops, vol_record)
    return [ch1_data, ch2_data,
            time, vol_start, vol_stim, vol_img]
