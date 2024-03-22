#!/usr/bin/env python3

import gc
import os
import h5py
from datetime import datetime
from suite2p import extraction_wrapper


# extract fluorescence signals from masks given as stat file.

def get_fluorescence(
        ops,
        stat,
        f_reg_ch1,
        f_reg_ch2
        ):
    stat, F_ch1, Fneu_ch1, F_ch2, Fneu_ch2 = extraction_wrapper(
        stat = stat,
        f_reg = f_reg_ch1,
        f_reg_chan2 = f_reg_ch2,
        ops = ops)
    fluo = [F_ch1, F_ch2][ops['functional_chan']-1]
    neuropil = [Fneu_ch1, Fneu_ch2][ops['functional_chan']-1]
    return [stat, fluo, neuropil]


# save the trace data.

def save_traces(
        ops,
        fluo, neuropil,
        ):
    # file structure:
    # ops['save_path0'] / raw_traces.h5
    # -- raw
    # ---- fluo
    # ---- neuropil
    f = h5py.File(os.path.join(
        ops['save_path0'], 'raw_traces.h5'), 'w')
    dict_group = f.create_group('raw')
    dict_group['fluo'] = fluo
    dict_group['neuropil'] = neuropil
    f.close()


# delete registration binary files.

def clean_reg_bin(
        ops,
        f_reg_ch1, f_reg_ch2
        ):
    try:
        del f_reg_ch1
        del f_reg_ch2
        gc.collect()
        os.remove(os.path.join(ops['save_path0'], 'temp', 'reg_ch1.bin'))
        os.remove(os.path.join(ops['save_path0'], 'temp', 'reg_ch2.bin'))
    except:
        None
        

# main function for fluorescence signal extraction from ROIs.

def run(ops, stat_func, f_reg_ch1, f_reg_ch2):
    print('===============================================')
    print('======= extracting fluorescence signals =======')
    print('===============================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    [stat, fluo, neuropil] = get_fluorescence(
         ops,
         stat_func,
         f_reg_ch1,
         f_reg_ch2)
    print('Fluorescence extraction completed')

    save_traces(ops, fluo, neuropil)
    print('Traces data saved')

    clean_reg_bin(ops, f_reg_ch1, f_reg_ch2)

