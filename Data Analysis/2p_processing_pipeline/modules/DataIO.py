#!/usr/bin/env python3

import os
import tifffile
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime


def list_filenames(
        ops
        ):
    ch1_files = [f for f in os.listdir(ops['data_path']) if "Ch1" in f]
    ch2_files = [f for f in os.listdir(ops['data_path']) if "Ch2" in f]
    vol_record = [f for f in os.listdir(ops['data_path']) if "VoltageRecording" in f]
    ch1_files.sort()
    ch2_files.sort()
    return ch1_files, ch2_files, vol_record


def read_tif_to_np(
        ops,
        ch_files
        ):
    ch_data = []
    for f in tqdm(ch_files):
        data = tifffile.imread(os.path.join(ops['data_path'], f))
        ch_data.append(data)
    ch_data = np.concatenate(ch_data, axis=0)
    ch_data = ch_data.astype('float32')
    return ch_data


def read_vol_to_np(
        ops,
        vol_record
        ):
    df_vol = pd.read_csv(
        os.path.join(ops['data_path'],vol_record[0]))
    time      = df_vol.iloc[:,0].to_numpy()
    vol_start = df_vol.iloc[:,1].to_numpy()
    vol_stim  = df_vol.iloc[:,2].to_numpy()
    vol_img   = df_vol.iloc[:,4].to_numpy()
    return time, vol_start, vol_stim, vol_img


def run(ops):
    print('===============================================')
    print('========== read and merge video data ==========')
    print('===============================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ch1_files, ch2_files, vol_record = list_filenames(ops)
    print('Found {} files for channel 1'.format(len(ch1_files)))
    print('Found {} files for channel 2'.format(len(ch2_files)))
    print('Reading channel 1 data')
    ch1_data = read_tif_to_np(ops, ch1_files)
    print('Reading channel 2 data')
    ch2_data = read_tif_to_np(ops, ch2_files)
    print('Reading voltage recordings')
    time, vol_start, vol_stim, vol_img = read_vol_to_np(ops, vol_record)
    return [ch1_data, ch2_data,
            time, vol_start, vol_stim, vol_img]
