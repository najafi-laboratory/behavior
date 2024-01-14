#!/usr/bin/env python3

import os
from suite2p.registration import register
from suite2p.io import BinaryFile


def create_file_to_reg(
        ops,
        r_ch_data,
        g_ch_data
        ):
    f_reg_ch1 = BinaryFile(
        Ly=512,
        Lx=512,
        filename=os.path.join(ops['save_path0'], 'reg_ch1.bin'),
        n_frames=r_ch_data.shape[0])
    f_reg_ch2 = BinaryFile(
        Ly=512,
        Lx=512,
        filename=os.path.join(ops['save_path0'], 'reg_ch2.bin'),
        n_frames=g_ch_data.shape[0])
    return f_reg_ch1, f_reg_ch2


def run(ops, r_ch_data, g_ch_data):
    print('===============================================')
    print('============= registering imaging =============')
    print('===============================================')
    f_reg_ch1, f_reg_ch2 = create_file_to_reg(ops, r_ch_data, g_ch_data)
    print('Registered channel files created in {}'.format(ops['save_path0']))
    reg_ref, _, _, \
    mean_ch1, _, _, _, \
    mean_ch2, _, _, _ = register.registration_wrapper(
        f_reg=f_reg_ch1,
        f_raw=r_ch_data,
        f_reg_chan2=f_reg_ch2,
        f_raw_chan2=g_ch_data,
        ops=ops)
    return [f_reg_ch1, f_reg_ch2, reg_ref, mean_ch1, mean_ch2]

