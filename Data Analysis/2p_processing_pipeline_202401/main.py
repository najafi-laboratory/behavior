#!/usr/bin/env python3

import argparse

from modules import Params
from modules import DataIO
from modules import MoveCorrect
from modules import Detection
from modules import Extraction
from modules import Visualize

'''
python main.py `
--denoise 1 `
--spatial_scale 1 `
--data_path './testdata/P3' `
--save_path0 './results/P3' `
--nchannels 2 `
--functional_chan 2 `
--brain_region 'ppc' `
'''

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Experiments can go shit but Yicong will love you forever!')
    parser.add_argument('--denoise',         required=True, type=int, help='Whether run denoising algorithm.')
    parser.add_argument('--spatial_scale',   required=True, type=int, help='The optimal scale in suite2p.')
    parser.add_argument('--data_path',       required=True, type=str, help='Path to the 2P imaging data.')
    parser.add_argument('--save_path0',      required=True, type=str, help='Path to save the results.')
    parser.add_argument('--nchannels',       required=True, type=int, help='Specify the number of channels.')
    parser.add_argument('--functional_chan', required=True, type=int, help='Specify functional channel id.')
    parser.add_argument('--brain_region',    required=True, type=str, help='Can only be crbl or ppc.')
    args = parser.parse_args()

    # parameters.
    ops = Params.run(args)

    # read video data.
    [ch1_data, ch2_data] = DataIO.run(ops)

    # motion correction.
    [f_reg_ch1, f_reg_ch2] = MoveCorrect.run(
         ops, ch1_data, ch2_data)

    # ROI detection.
    stat_func = Detection.run(
        ops, f_reg_ch1, f_reg_ch2)

    # Signal extraction.
    Extraction.run(
        ops,
        stat_func,
        f_reg_ch1,
        f_reg_ch2)
    
    # plot figures
    Visualize.run(ops)