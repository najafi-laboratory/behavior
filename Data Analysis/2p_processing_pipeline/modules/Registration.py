#!/usr/bin/env python3

import os
import h5py
import numpy as np
from datetime import datetime
from suite2p.registration import register
from suite2p.io import BinaryFile


# make new file to save registered data.
def create_file_to_reg(
        ops,
        ch1_data,
        ch2_data
        ):
    # create the temp folder if not exist.
    if not os.path.exists(os.path.join(ops['save_path0'], 'temp')):
        os.makedirs(os.path.join(ops['save_path0'], 'temp'))
    # ch1 binary file.
    f_reg_ch1 = BinaryFile(
        Ly=ops['Ly'],
        Lx=ops['Lx'],
        filename=os.path.join(ops['save_path0'], 'temp', 'reg_ch1.bin'),
        n_frames=ch1_data.shape[0])
    # ch2 binary file.
    f_reg_ch2 = BinaryFile(
        Ly=ops['Ly'],
        Lx=ops['Lx'],
        filename=os.path.join(ops['save_path0'], 'temp', 'reg_ch2.bin'),
        n_frames=ch2_data.shape[0])
    return f_reg_ch1, f_reg_ch2


# compute max proj image.
def get_proj_img(
        f_reg_ch1,
        f_reg_ch2
        ):
    max_ch1 = np.max(f_reg_ch1.data, axis=0)
    max_ch2 = np.max(f_reg_ch2.data, axis=0)
    return max_ch1, max_ch2


# save the mean and max image to h5 file in temp.
def save_proj_img(
        ops,
        reg_ref,
        mean_ch1, mean_ch2,
        max_ch1,  max_ch2
        ):
    # file structure:
    # ops['save_path0'] / temp / proj_img.h5
    # -- proj_img
    # ---- reg_ref
    # ---- mean_ch1
    # ---- mean_ch2
    # ---- max_ch1
    # ---- max_ch2
    f = h5py.File(os.path.join(
        ops['save_path0'], 'temp', 'proj_img.h5'), 'w')
    grp = f.create_group('proj_img')
    grp['reg_ref'] = reg_ref
    grp['mean_ch1'] = mean_ch1
    grp['mean_ch2'] = mean_ch2
    grp['max_ch1'] = max_ch1
    grp['max_ch2'] = max_ch2
    f.close()


# main function for registration.
def run(ops, ch1_data, ch2_data):
    print('===============================================')
    print('============= registering imaging =============')
    print('===============================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # create registration files.
    f_reg_ch1, f_reg_ch2 = create_file_to_reg(ops, ch1_data, ch2_data)
    print('Registered channel files created in {}.'.format(ops['save_path0']))
    # suite2p registration.
    # reference image computation included.
    # motion correction included.
    reg_ref, _, _, mean_ch1, _, _, _, mean_ch2, _, _, _ = register.registration_wrapper(
        f_reg=f_reg_ch1,
        f_raw=ch1_data,
        f_reg_chan2=f_reg_ch2,
        f_raw_chan2=ch2_data,
        ops=ops)
    print('Registration completed.')
    # compute mean and max projection image.
    print('Computing max projection images.')
    max_ch1, max_ch2 = get_proj_img(f_reg_ch1, f_reg_ch2)
    # save projection and reference images.
    save_proj_img(
        ops,
        reg_ref,
        mean_ch1, mean_ch2,
        max_ch1,  max_ch2)
    print('Projected images saved.')
    return [f_reg_ch1, f_reg_ch2]

