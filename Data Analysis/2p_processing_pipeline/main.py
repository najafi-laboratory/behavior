#!/usr/bin/env python3

from modules import Params
from modules import DataIO
from modules import Registration
from modules import CellDetect
from modules import Extraction
from modules import SyncSignal


def run_pipeline(
        run_Registration = True,
        run_CellDetect = True,
        run_Extraction = True,
        run_SyncSignal = True,
        ):
    ops = Params.run()
    if run_Registration:
        [ch1_data, ch2_data,
         time,
         vol_start, vol_stim, vol_img] = DataIO.run(
             ops)
        [f_reg_ch1, f_reg_ch2] = Registration.run(
             ops,
             ch1_data, ch2_data)
    if run_CellDetect:
        [stat_ref, stat_ch1, stat_ch2] = CellDetect.run(
             ops)
    if run_Extraction:
        [] = Extraction.run(
             ops,
             stat_ref,
             f_reg_ch1,
             f_reg_ch2)
    if run_SyncSignal:
        [neural_trial] = SyncSignal.run(
            ops,
            time,
            vol_start, vol_stim, vol_img)


if __name__ == "__main__":

    run_pipeline()
