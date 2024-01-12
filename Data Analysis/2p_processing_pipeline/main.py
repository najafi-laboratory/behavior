from modules import Params
from modules import DataIO
from modules import Registration
from modules import CellDetect


def run():
    0



if __name__ == "__main__":
    run()

    ops = Params.run()

    [r_ch_data,
     g_ch_data] = DataIO.run(
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