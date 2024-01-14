#!/usr/bin/env python3

from modules import Params
from modules import DataIO
from modules import Registration
from modules import CellDetect
from modules import Extraction
from modules import SyncSignal


if __name__ == "__main__":

    ops = Params.run()

    [r_ch_data,
     g_ch_data,
     time,
     vol_start,
     vol_stim,
     vol_img] = DataIO.run(
         ops)

    [f_reg_ch1,
     f_reg_ch2,
     reg_ref,
     mean_ch1,
     mean_ch2] = Registration.run(
         ops,
         r_ch_data,
         g_ch_data)

    [stat_ref,
     stat_r_ch,
     stat_g_ch] = CellDetect.run(
         ops,
         reg_ref,
         mean_ch1,
         mean_ch2)

    [fluo_ch1,
     mean_fluo_ch1,
     spikes_ch1,
     fluo_ch2,
     mean_fluo_ch2,
     spikes_ch2] = Extraction.run(
         ops,
         stat_ref,
         f_reg_ch1,
         f_reg_ch2)

    [time_img,
     trial_stim,
     trial_fluo_ch1,
     trial_fluo_ch2,
     trial_mean_fluo_ch1,
     trial_mean_fluo_ch2,
     trial_spikes_ch1,
     trial_spikes_ch2] = SyncSignal.run(
             ops,
             time, vol_start, vol_stim, vol_img,
             fluo_ch1, mean_fluo_ch1, spikes_ch1,
             fluo_ch2, mean_fluo_ch2, spikes_ch2
             )