#!/usr/bin/env python3

import os
import h5py
import numpy as np
from datetime import datetime
from suite2p.registration import register
from suite2p.io import BinaryFile


def create_file_to_reg(
        ops,
        ch1_data,
        ch2_data
        ):
    if not os.path.exists(os.path.join(ops['save_path0'], 'temp')):
        os.makedirs(os.path.join(ops['save_path0'], 'temp'))
    f_reg_ch1 = BinaryFile(
        Ly=ops['Ly'],
        Lx=ops['Lx'],
        filename=os.path.join(ops['save_path0'], 'temp', 'reg_ch1.bin'),
        n_frames=ch1_data.shape[0])
    f_reg_ch2 = BinaryFile(
        Ly=ops['Ly'],
        Lx=ops['Lx'],
        filename=os.path.join(ops['save_path0'], 'temp', 'reg_ch2.bin'),
        n_frames=ch2_data.shape[0])
    return f_reg_ch1, f_reg_ch2


def get_proj_img(
        ch1_data,
        ch2_data
        ):
    mean_ch1 = np.mean(ch1_data, axis=0)
    mean_ch2 = np.mean(ch2_data, axis=0)
    max_ch1 = np.max(ch1_data, axis=0)
    max_ch2 = np.max(ch2_data, axis=0)
    return mean_ch1, mean_ch2, max_ch1, max_ch2


def save_proj_img(
        ops,
        reg_ref,
        mean_ch1, mean_ch2,
        max_ch1,  max_ch2
        ):
    f = h5py.File(os.path.join(
        ops['save_path0'], 'temp', 'proj_img.h5'), 'w')
    grp = f.create_group('proj_img')
    grp['reg_ref'] = reg_ref
    grp['mean_ch1'] = mean_ch1
    grp['mean_ch2'] = mean_ch2
    grp['max_ch1'] = max_ch1
    grp['max_ch2'] = max_ch2
    f.close()


def run(ops, ch1_data, ch2_data):
    print('===============================================')
    print('============= registering imaging =============')
    print('===============================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    mean_ch1, mean_ch2, max_ch1, max_ch2 = get_proj_img(ch1_data, ch2_data)
    f_reg_ch1, f_reg_ch2 = create_file_to_reg(ops, ch1_data, ch2_data)
    print('Registered channel files created in {}'.format(ops['save_path0']))
    reg_ref, _, _, _, _, _, _, _, _, _, _ = register.registration_wrapper(
        f_reg=f_reg_ch1,
        f_raw=ch1_data,
        f_reg_chan2=f_reg_ch2,
        f_raw_chan2=ch2_data,
        ops=ops)
    save_proj_img(
        ops,
        reg_ref,
        mean_ch1, mean_ch2,
        max_ch1,  max_ch2)
    print('Projected images saved')
    return [f_reg_ch1, f_reg_ch2]
