#!/usr/bin/env python3

import os
import tifffile
import numpy as np
import pandas as pd


def list_filenames(
        ops
        ):
    r_ch_files = [f for f in os.listdir(ops['data_path']) if "Ch1" in f]
    g_ch_files = [f for f in os.listdir(ops['data_path']) if "Ch2" in f]
    vol_record = [f for f in os.listdir(ops['data_path']) if "VoltageRecording" in f]
    r_ch_files.sort()
    g_ch_files.sort()
    return r_ch_files, g_ch_files, vol_record


def read_tif_to_np(
        ops,
        ch_files
        ):
    ch_data = []
    for f in ch_files:
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
    r_ch_files, g_ch_files, vol_record = list_filenames(ops)
    print('Found {} files for red channel'.format(len(r_ch_files)))
    print('Found {} files for green channel'.format(len(g_ch_files)))
    print('Reading red channel data')
    r_ch_data = read_tif_to_np(ops, r_ch_files)
    print('Reading green channel data')
    g_ch_data = read_tif_to_np(ops, g_ch_files)
    print('Reading voltage recordings')
    time, vol_start, vol_stim, vol_img = read_vol_to_np(ops, vol_record)
    return [r_ch_data, g_ch_data,
            time, vol_start, vol_stim, vol_img]
