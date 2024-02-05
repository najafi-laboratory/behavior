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
from plot.fig4_align_grating import plot_fig4
from plot.fig5_align_error import plot_fig5

'''
python main.py `
--run_Registration 1 `
--run_CellDetect 1 `
--run_Extraction 1 `
--run_SyncSignal 1 `
--run_RetrieveResults 1 `
--run_Plotter 1 `
--data_path './testdata/FN8_P_Omii_020224_-2730_1285_-85_debug-272' `
--save_path0 './results/test_omi' `
--nchannels 2 `
--functional_chan 2 `
--diameter 8
'''
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Before using the code say I Love Yicong Forever very loudly!')
    parser.add_argument('--run_Registration',    type=int, default=1, help='Whether run the registration module.')
    parser.add_argument('--run_CellDetect',      type=int, default=1, help='Whether run the cell detection module.')
    parser.add_argument('--run_Extraction',      type=int, default=1, help='Whether run the signal extraction module.')
    parser.add_argument('--run_SyncSignal',      type=int, default=1, help='Whether run the synchronization module.')
    parser.add_argument('--run_RetrieveResults', type=int, default=1, help='Whether run the data retrieval module.')
    parser.add_argument('--run_Plotter',         type=int, default=1, help='Whether plot the results.')
    parser.add_argument('--data_path',       required=True, type=str, help='Path to the 2P imaging data.')
    parser.add_argument('--save_path0',      required=True, type=str, help='Path to save the results.')
    parser.add_argument('--nchannels',       required=True, type=int, help='Specify the number of channels.')
    parser.add_argument('--functional_chan', required=True, type=int, help='Specify functional channel id.')
    parser.add_argument('--diameter',        required=True, type=int, help='Cell diameter for cellpose detection.')
    args = parser.parse_args()

    # parameters.
    ops = Params.run(args)
    
    # read video data.
    [ch1_data, ch2_data] = DataIO.run(ops)

    # registration.
    if args.run_Registration:
        [f_reg_ch1, f_reg_ch2] = Registration.run(
             ops, ch1_data, ch2_data)

    # ROI detection.
    if args.run_CellDetect:
        [stat_func] = CellDetect.run(ops)

    # Signal extraction.
    if args.run_Extraction:
        Extraction.run(
            ops,
            stat_func,
            f_reg_ch1,
            f_reg_ch2)

    # Synchronized signals.
    if args.run_SyncSignal:
        SyncSignal.run(ops)

    # read processed data.
    if args.run_RetrieveResults:
        [mask,
         raw_traces,
         raw_voltages,
         neural_trials] = RetrieveResults.run(ops)

    # plot results
    if args.run_Plotter:
        plot_fig1(ops)
        plot_fig2(ops)
        plot_fig3(ops)
        plot_fig4(ops)
        plot_fig5(ops)


