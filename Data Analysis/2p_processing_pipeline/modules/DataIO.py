#!/usr/bin/env python3

import gc
import os
import tifffile
import numpy as np
from tqdm import tqdm
from datetime import datetime


# find the target filenames in ops['data_path'].

def list_filenames(
        ops
        ):
    # ch1: SESSION_Cycle00001_Ch1_000NUM.ome.tif.
    # ch2: SESSION_Cycle00001_Ch2_000NUM.ome.tif.
    ch1_files = [f for f in os.listdir(ops['data_path'])
                 if 'Ch1' in f and '.tif' in f]
    ch2_files = [f for f in os.listdir(ops['data_path'])
                 if 'Ch2' in f and '.tif' in f]
    # sort the filename based on the recording partition number.
    ch1_files.sort()
    ch2_files.sort()
    return ch1_files, ch2_files


# read the tif files given the filename list.

def read_tif_to_np(
        ops,
        ch_files
        ):
    # no channel data found.
    if len(ch_files) == 0:
        ch_data = np.zeros((1, ops['Lx'], ops['Ly']), dtype='float32')
    # read into array.
    else:
        ch_data = np.empty((0, ops['Lx'], ops['Ly']), dtype='float32')
        for f in tqdm(ch_files):
            data = tifffile.imread(os.path.join(ops['data_path'], f))
            ch_data = np.concatenate((ch_data, data), axis=0)
            # release memory.
            del data
            gc.collect()
    return ch_data


# main function for reading the data.

def run(ops):

    print('===============================================')
    print('========== read and merge video data ==========')
    print('===============================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    ch1_files, ch2_files = list_filenames(ops)
    print('Found {} files for channel 1'.format(len(ch1_files)))
    print('Found {} files for channel 2'.format(len(ch2_files)))

    print('Reading channel 1 data')
    print(ch1_files)
    ch1_data = read_tif_to_np(ops, ch1_files)

    print('Reading channel 2 data')
    print(ch2_files)
    ch2_data = read_tif_to_np(ops, ch2_files)

    return [ch1_data, ch2_data]
