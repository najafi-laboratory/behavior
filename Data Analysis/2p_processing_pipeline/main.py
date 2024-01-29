#!/usr/bin/env python3


import argparse

from modules import Params
from modules import DataIO
from modules import Registration
from modules import CellDetect
from modules import Extraction
from modules import SyncSignal
from modules import RetrieveResults

from plot.fig1_mask import plot_fig1
from plot.fig2_stim_distribution import plot_fig2
from plot.fig3_raw_traces import plot_fig3


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='I Love Yicong Forever.')
    parser.add_argument('--run_Registration',    type=bool, default=True, help='Whether run the registration module.')
    parser.add_argument('--run_CellDetect',      type=bool, default=True, help='Whether run the cell detection module.')
    parser.add_argument('--run_Extraction',      type=bool, default=True, help='Whether run the signal extraction module.')
    parser.add_argument('--run_SyncSignal',      type=bool, default=True, help='Whether run the synchronization module.')
    parser.add_argument('--run_RetrieveResults', type=bool, default=True, help='Whether run the data retrieval module.')
    parser.add_argument('--run_Plotter',         type=bool, default=True, help='Whether plot the results.')
    parser.add_argument('--data_path',       required=True, type=str, help='Path to the 2P imaging data.')
    parser.add_argument('--save_path0',      required=True, type=str, help='Path to save the results.')
    parser.add_argument('--functional_chan', required=True, type=int, help='Specify functional channel id.')
    args = parser.parse_args()
    
    # parameters.
    ops = Params.run(args)
    
    # registration.
    if args.run_Registration:
        [ch1_data, ch2_data,
         time,
         vol_start, vol_stim, vol_img] = DataIO.run(
             ops)
        [f_reg_ch1, f_reg_ch2] = Registration.run(
             ops,
             ch1_data, ch2_data)
    
    # ROI detection.
    if args.run_CellDetect:
        [stat_ref, _, _] = CellDetect.run(
             ops)
    
    # Signal extraction.
    if args.run_Extraction:
        [] = Extraction.run(
             ops,
             stat_ref,
             f_reg_ch1,
             f_reg_ch2)
    
    # Synchronized signals.
    if args.run_SyncSignal:
        [] = SyncSignal.run(
            ops,
            time,
            vol_start, vol_stim, vol_img)
    
    # read processed data.
    if args.run_RetrieveResults:
        [neural_trial, mask] = RetrieveResults.run(
            ops)

    # plot results
    if args.run_Plotter:
        plot_fig1(ops)
        plot_fig2(ops)
        plot_fig3(ops)

        
        
        
        