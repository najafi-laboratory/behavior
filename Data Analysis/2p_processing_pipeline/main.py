#!/usr/bin/env python3

import argparse

from modules import Params
from modules import DataIO
from modules import Registration
from modules import Detection
from modules import Extraction
from modules import PostProcess

from plot.fig1_mask import plot_fig1
from plot.fig2_raw_traces import plot_fig2
from plot.fig4_align_grating import plot_fig4
from plot.fig5_align_error import plot_fig5
from plot.fig6_spike_trigger_average import plot_fig6


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Experiments can go shit but Yicong will love you forever!')
    parser.add_argument('--run_Plotter',     required=True, type=int, help='Whether plot the results.')
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

    # registration.
    [f_reg_ch1, f_reg_ch2] = Registration.run(
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

    # Synchronized signals.
    PostProcess.run_vol(ops)

    # plot results
    if args.run_Plotter:
        plot_fig1(ops)
        plot_fig2(ops)
        plot_fig4(ops)
        plot_fig5(ops)
        plot_fig6(ops)


