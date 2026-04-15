import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')

import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime

from Module.Reader import load_mat
from Module.session_name_parse import parse_behavior_file_path
from Module.Trial_properties import extract_session_properties
from Module.Licking_properties import extract_lick_properties

from plotter.session_structure import plot_trial_outcome_vs_type_with_isi
from plotter.Outcome import plot_all_sessions_with_gridspec
from plotter.opto_sequence_performence import plot_opto_seq_with_gridspec
from plotter.licking import (
    plot_psychometric_curve, plot_pooled_psychometric_curve, plot_grand_average_psychometric_curve,
    plot_isi_reaction_time, plot_pooled_isi_reaction_time, plot_grand_average_isi_reaction_time,
    plot_reaction_time_curve, plot_pooled_reaction_time_curve, plot_grand_average_reaction_time_curve
)

from plotter.licking_less_detailed import (
    plot_psychometric_curve_less, plot_pooled_psychometric_curve_less, plot_grand_average_psychometric_curve_less,
    plot_isi_reaction_time_less, plot_pooled_isi_reaction_time_less, plot_grand_average_isi_reaction_time_less,
    plot_reaction_time_curve_less, plot_pooled_reaction_time_curve_less, plot_grand_average_reaction_time_curve_less
)

from plotter.psychometric_halves import (
    plot_psychometric_neutral_and_halves, plot_pooled_psychometric_neutral_and_halves, plot_grand_average_psychometric_neutral_and_halves
    )
from plotter.Opto_sequence_psychometric import (
    plot_opto_comparison_curve, plot_pooled_opto_comparison_curve, plot_grand_average_opto_comparison_curve,
    plot_opto_comparison_reaction_time_curve, plot_pooled_opto_comparison_reaction_time_curve, plot_grand_average_opto_comparison_reaction_time_curve
)

from plotter.rare_trial_analysis import plot_all_rare_trial_analyses
from plotter.Performance_by_stimulus import plot_performance_by_stimulus
from plotter.learned_trial_number_expo_fitted import analyze_learning_dynamics_fitted
from plotter.learned_trial_number import analyze_learning_dynamics
from plotter.Reaction_time_rare_trials import plot_reaction_time_rare_trials
from plotter.Reaction_time_block_transition import plot_reaction_time_transitions
from plotter.Reaction_time_stimulus_type import plot_rt_by_block_status
from plotter.Adaptaion_around_rare import plot_rare_trial_performance
from plotter.trial_by_trial_adaptation import plot_block_transitions
from plotter.psychometric_epoch import plot_psychometric_epochs
from plotter.Majority_trials_analysis import plot_all_majority_trial_analysis
from plotter.Rare_vs_majority import plot_all_rare_vs_majority_analysis 
from plotter.Post_rare_prior_update import plot_post_rare_trial_analysis
from plotter.psychometric_Blockwise import plot_psychometric_control_opto_epochs
from plotter.Summary_stat import plot_summary_stats

# Configuration
# YH24LG ##############################################################################################################################
# NOTE: These are the sessions we started double block paradigm
# NOTE: Sessions 20250714, 20250801, 20250905, 20250916, 20250919, 20251114 are excluded 
# DATA_PATHS = [
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250624_150012.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250625_164242.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250626_182737.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250628_192014.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250630_174909.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250701_154950.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250702_140050.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250703_164706.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250708_192901.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250709_174856.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250710_130625.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250711_093117.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250715_180125.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250717_144725.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250718_172810.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250722_175504.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250730_193455.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250731_163425.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250805_133504.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250808_180200.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250811_195235.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250814_121831.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250819_182316.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250822_143100.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250826_143546.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250902_155717.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250903_112248.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250904_145410.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250909_155820.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250910_163253.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250911_143000.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250912_141501.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250917_215445.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250923_163844.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250924_180806.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250925_173549.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250929_160056.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20250930_164710.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20251001_151717.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20251002_154335.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20251003_151814.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20251008_173017.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20251009_161811.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20251010_171101.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20251013_170824.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20251014_150532.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20251015_164533.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_block_single_interval_discrimination_V_1_20251016_151502.mat",
#               ]


# SAVE_PATH = 'F:\\Single_Interval_discrimination\\Figures\\YH24LG'

# MC06_SChR ##############################################################################################################################

DATA_PATHS = [
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250708_170047.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250709_181730.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250718_141657.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250721_151425.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250723_121759.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250728_144311.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250730_180438.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250801_144807.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250805_171451.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250806_173317.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250807_184602.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250808_121500.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250811_170819.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250815_134127.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250819_110233.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250902_191953.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250903_134632.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250904_112921.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250908_151528.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250909_145508.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250910_144110.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250912_143347.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250915_155646.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250923_140941.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250924_151234.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250926_143659.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250929_145231.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250930_145534.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251001_174232.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251002_135323.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251003_144924.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251007_132950.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251008_160423.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251009_150137.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251010_142810.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251013_161935.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251014_155647.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251015_155010.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251016_124808.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251020_171536.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251021_140348.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251022_130756.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251023_141919.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251024_153832.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251027_190105.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251028_111727.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251029_131423.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251030_140422.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251031_141541.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251103_141954.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251104_144535.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251105_110149.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251106_145814.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251107_131242.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251110_160243.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251111_143458.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251113_141100.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251124_155049.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251125_154612.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251126_110859.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251128_083211.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251202_122724.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251204_143218.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251205_135637.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251206_151925.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251209_145044.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251210_101224.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251211_145503.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251212_134446.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251215_114901.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251216_151339.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251217_150520.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251218_164007.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251222_134238.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260108_130953.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260109_132420.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260112_183827.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260113_095249.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260114_144555.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260115_154411.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260120_132752.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260121_143350.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260122_150723.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260123_110027.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260127_134849.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260128_134756.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260129_095204.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260130_111626.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260202_143727.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260203_140921.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260204_143807.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260205_162549.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260209_110037.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260210_152859.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260211_132110.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260212_142543.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260213_105123.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260217_171315.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260218_162836.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260219_135617.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260220_110558.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260224_160017.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260225_135329.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260226_133633.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260227_115418.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260302_140324.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260303_152200.mat",
              "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260304_150642.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260305_150431.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260306_133055.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260309_152730.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260310_164042.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260311_135102.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260312_150454.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260313_154230.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260316_143647.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260317_121120.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260318_160047.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260319_140601.mat",
            #   "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20260320_102546.mat",

             ]

# non-opto:
    # Mid 2: 07/21, 08/01, 09/29, 09/30, 12/09, 12/10
    # Well- trained: 11/24, 11/25, 11/26, 11/28, 12/02, 12/04, 12/05, 12/06, 12/11
    # NOTE: Sessions 11/24, 11/25, 11/26, 11/28 are excluded
# DATA_PATHS = ["F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250721_151425.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250801_144807.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250929_145231.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250930_145534.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251103_141954.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251104_144535.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251105_110149.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251106_145814.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251107_131242.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251110_160243.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251202_122724.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251204_143218.mat",
#              ]


# random opto:
    # Mid 2 (80/20, before the trial fraction of each block is 70/30): 09/03, 09/04, 09/09, 09/10, 09/12
    # NOTE: Session 11/25, 09/12 is excluded
# DATA_PATHS = [
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250903_134632.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250904_112921.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250909_145508.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250910_144110.mat",
#              ]


# Early-block opto:
    # Mid 2: 09/15, 09/23, 09/24, 10/02, 10/03, 10/07, 10/08, 10/09, 10/10, 10/13, 10/14, 10/21
    # Well-trained: 10/15, 10/16, 10/22, 10/23, 10/24, 10/27, 10/28, 10/29, 10/30, 10/31, 11/11 (edited) 
    # NOTE: Sessions 10/03, 10/24, 10/28 is excluded
# DATA_PATHS = [
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250915_155646.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250923_140941.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20250924_151234.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251002_135323.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251007_132950.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251008_160423.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251009_150137.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251010_142810.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251013_161935.mat",    
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251014_155647.mat",    
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251021_140348.mat",    
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251015_155010.mat",    
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251016_124808.mat",    
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251022_130756.mat",    
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251023_141919.mat",       
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251027_190105.mat",    
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251029_131423.mat", 
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251031_141541.mat", 
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC06\MC06_SChR2_block_single_interval_discrimination_V_1_20251111_143458.mat",            
#              ]

SAVE_PATH = 'F:\\Single_Interval_discrimination\\Figures\\MC06_SChR'

# MC07_SChR ##############################################################################################################################
# non-opto:
    # Mid 2: 08/01, 09/08, 09/26, 09/29, 09/30, 10/01
    # Well- trained: 11/24, 11/25, 11/26, 11/28, 12/02, 12/04
    # NOTE: Sessions 09/28, 09/26, 10/01, 11/10, 11/25, 11/26 are excluded
# DATA_PATHS = [
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20250801_160230.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20250929_162847.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251103_151600.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251104_153124.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251105_153946.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251106_155500.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251107_141717.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251124_163510.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251128_091455.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251202_131606.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251204_152940.mat",
#               ]


# random opto:
    # Mid 2: 09/03, 09/08, 09/09, 09/10
    # NOTE: Session 09/03, 09/08, 09/09, 09/10 is excluded
# DATA_PATHS = [
#               
#              ]


# Early-block opto:
    # Mid 2: 09/15, 10/01, 10/02, 10/03, 10/07, 10/08, 10/09, 10/10, 10/13, 10/14, 10/15, 10/21, 10/22
    # Well-trained: 10/27, 10/28, 10/29, 10/30, 10/31,11/03, 11/04, 11/05, 11/06, 11/07, 11/10, 11/11
# DATA_PATHS = [
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20250915_164829.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251001_163659.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251002_145135.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251003_155130.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251007_154027.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251008_173530.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251009_163015.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251010_154737.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251013_151852.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251014_165705.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251015_145514.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251021_145908.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251022_141742.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251027_180852.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251028_133131.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251029_141932.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251030_162648.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251031_151428.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC07\MC07_SChR2_block_single_interval_discrimination_V_1_20251111_152919.mat",
#               ]

# SAVE_PATH = 'F:\\Single_Interval_discrimination\\Figures\\MC07_SChR'

# MC08_SChR ##############################################################################################################################
# non-opto:
    # Mid 2: 08/07, 08/08, 08/11, 08/12, 08/14, 08/15, 08/19, 08/20, 08/21, 08/22, 08/25
    # Well- trained: 11/25, 11/26, 11/28, 12/02, 12/03, 12/04, 12/05, 12/09, 12/10, 12/11
    # NOTE: Sessions 08/22, 11/25, 1203 are excluded
# DATA_PATHS = [
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250807_161922.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250808_141934.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250811_190814.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250812_155927.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250814_113936.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250815_165438.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250819_172012.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250820_130848.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250821_143112.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250825_144118.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251103_161207.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251105_164015.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251126_153003.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251128_095657.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251202_140232.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251204_161155.mat",
#               ]


# random opto:
    # Mid 2 (70/30): 07/31, 08/05, 08/06, 08/26, 09/02,
    # Mid 2 (80/20): 09/03, 09/04, 09/08, 09/09, 09/10, 09/12
    # NOTE: Sessions 07/31 are excluded
# DATA_PATHS = [
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250805_105209.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250806_153404.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250826_133754.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250902_172446.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250903_113809.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250904_153648.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250908_160433.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250909_172047.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250910_165937.mat",
#               ]    


# Early-block opto:
    # Mid 2: 09/15, 09/17, 09/22, 09/23, 09/24, 09/25, 09/26, 09/29, 09/30, 10/01, 10/02, 10/03, 10/07, 10/08, 10/09, 10/10, 10/13, 10/14
    # Well-trained: 10/15, 10/16, 10/21, 10/22, 10/23, 10/27, 10/28, 10/29, 10/30, 10/31, 11/04, 11/06, 11/07, 11/10, 11/11
    # NOTE: Sessions 09/17, 9/23, 10/09, 10/10, 10/23, 10/31, 11/06, 11/07, 11/11 are excluded
# DATA_PATHS = [
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250915_145020.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250922_110520.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250924_140225.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250925_140419.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250926_135711.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250929_141221.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20250930_140955.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251001_152748.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251002_103634.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251003_134413.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251007_110204.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251008_150910.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251013_171746.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251014_143422.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251015_170000.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251016_152558.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251021_131128.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251022_151220.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251027_170933.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251028_143535.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251029_152156.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251030_151008.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251104_104812.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC08\MC08_SChR2_block_single_interval_discrimination_V_1_20251110_150812.mat",
#               ]

# SAVE_PATH = 'F:\\Single_Interval_discrimination\\Figures\\MC08_SChR'

# MC09_SChR ##############################################################################################################################
# non-opto: 
    # 07/31 was the only non-opto session using double-block protocol.



# random opto:
    # Mid 1: 07/07, 07/09, 07/11, 07/14, 07/15
    # Mid 2:07/16, 07/17, 07/18, 07/21, 07/22, 07/23, 07/24, 07/30, 07/31, 08/01, 08/05, 08/06, 08/07, 08/08, 08/11, 08/12, 08/15, 08/18, 08/19, 08/21, 08/25, 08/26
# DATA_PATHS = ["F:\Single_Interval_discrimination\Data_behavior\SCHR_MC09\MC09_SChR2_block_single_interval_discrimination_V_1_20240707_155808.mat",
#               "F:\Single_Interval_discrimination\Data_behavior\SCHR_MC09\MC09


# SAVE_PATH = 'F:\\Single_Interval_discrimination\\Figures\\MC09_SChR'
###############################################################################################################################################

# Ensure save directory exists
os.makedirs(SAVE_PATH, exist_ok=True)

def load_session_data(path):
    """Load and parse session data once per file."""
    session = load_mat(path)
    subject, version, date = parse_behavior_file_path(path)
    return {
        'session': session,
        'subject': subject,
        'version': version,
        'date': date,
        'trial_properties': extract_session_properties(session, subject, version, date),
        'lick_properties': extract_lick_properties(session, subject, version, date)
    }

def generate_session_plots_pdf(data_paths, save_path=SAVE_PATH):
    """Generate PDF with trial outcome plots for multiple sessions."""
    subject = parse_behavior_file_path(data_paths[0])[0]
    output_pdf = os.path.join(save_path, f'session_bpod_{subject}_{data_paths[-1].split("_")[-2]}_{data_paths[0].split("_")[-2]}.pdf')
    
    with PdfPages(output_pdf) as pdf:
        for path in data_paths:
            data = load_session_data(path)
            fig = plot_trial_outcome_vs_type_with_isi(data['trial_properties'])
            pdf.savefig(fig)
            plt.close(fig)
    
    return output_pdf

def prepare_session_data(data_paths):
    """Prepare session data for outcome and lick analysis."""
    sessions_data = {
        'dates': [], 'outcomes': [], 'opto_tags': [], 'trial_types': [],
        'lick_properties': [], 'block_type': []
    }
    
    for path in data_paths:
        data = load_session_data(path)
        sessions_data['dates'].append(data['date'])
        sessions_data['outcomes'].append(data['trial_properties']['outcome'])
        sessions_data['opto_tags'].append(data['trial_properties']['opto_tag'])
        sessions_data['trial_types'].append(data['trial_properties']['trial_type'])
        sessions_data['lick_properties'].append(data['lick_properties'])
        sessions_data['block_type'].append(data['trial_properties']['block_type'])
    
    return sessions_data

def plot_all_sessions_outcome(sessions_data, data_paths, save_path=SAVE_PATH):
    """Plot all sessions' outcomes."""
    subject = parse_behavior_file_path(data_paths[0])[0]
    fig = plot_all_sessions_with_gridspec(
        sessions_data['outcomes'], sessions_data['dates'],
        sessions_data['trial_types'], sessions_data['opto_tags'], sessions_data['block_type'],
        figsize=(40, 30)
    )
    
    name = f'outcome_{subject}_{data_paths[-1].split("_")[-2]}_{data_paths[0].split("_")[-2]}.pdf'
    output_path = os.path.join(save_path, name)
    fig.savefig(output_path)
    plt.close(fig)
    
    return name

def plot_all_sessions_opto_seq(sessions_data, data_paths, save_path=SAVE_PATH):
    """Plot all sessions' outcomes."""
    subject = parse_behavior_file_path(data_paths[0])[0]
    fig = plot_opto_seq_with_gridspec(
        sessions_data['outcomes'], sessions_data['dates'],
        sessions_data['trial_types'], sessions_data['opto_tags'],
        figsize=(40, 15)
    )
    
    name = f'Opto_seq_performance_{subject}_{data_paths[-1].split("_")[-2]}_{data_paths[0].split("_")[-2]}.pdf'
    output_path = os.path.join(save_path, name)
    fig.savefig(output_path)
    plt.close(fig)
    
    return name

def plot_psychometric_analysis(sessions_data, data_paths, save_path=SAVE_PATH):
    """Generate comprehensive psychometric analysis plots."""
    n_sessions = len(data_paths)
    subject = parse_behavior_file_path(data_paths[0])[0]
    
    # Dynamic figure size
    fig = plt.figure(figsize=(90, (2 + n_sessions) * 5))
    gs = fig.add_gridspec(2 + n_sessions, 12)
    
    # Plot configurations
    plot_configs = [
        ('psychometric', plot_psychometric_curve, plot_pooled_psychometric_curve, plot_grand_average_psychometric_curve),
        ('isi_reaction', plot_isi_reaction_time, plot_pooled_isi_reaction_time, plot_grand_average_isi_reaction_time),
        ('reaction_time', plot_reaction_time_curve, plot_pooled_reaction_time_curve, plot_grand_average_reaction_time_curve)
    ]
    
    # Outcome filters and settings
    outcomes = [('all', False), ('all', True), ('rewarded', True), ('punished', True)]
    
    for i, (plot_type, single_fn, pooled_fn, grand_fn) in enumerate(plot_configs):
        # Single session plots
        for j, session_data in enumerate(sessions_data['lick_properties']):
            for k, (outcome, opto_split) in enumerate(outcomes):
                if plot_type == 'psychometric' or (plot_type != 'psychometric' and opto_split):
                    ax = fig.add_subplot(gs[2 + j, k + (i * 4)])
                    single_fn(session_data, ax=ax, filter_outcomes=outcome, opto_split=opto_split)
        
        # Pooled plots
        for k, (outcome, opto_split) in enumerate(outcomes):
            if plot_type == 'psychometric' or (plot_type != 'psychometric' and opto_split):
                ax = fig.add_subplot(gs[0, k + (i * 4)])
                kwargs = {'bin_width': 0.05, 'fit_quadratic': True} if plot_type == 'reaction_time' else {}
                if plot_type == 'psychometric' and outcome == 'all' and not opto_split:
                    kwargs['fit_logistic'] = True
                pooled_fn(sessions_data['lick_properties'], ax=ax, filter_outcomes=outcome, opto_split=opto_split, **kwargs)
        
        # Grand average plots
        for k, (outcome, opto_split) in enumerate(outcomes):
            if plot_type == 'psychometric' or (plot_type != 'psychometric' and opto_split):
                ax = fig.add_subplot(gs[1, k + (i * 4)])
                kwargs = {'bin_width': 0.05, 'fit_quadratic': True} if plot_type == 'reaction_time' else {}
                if plot_type == 'psychometric':
                    kwargs['fit_logistic'] = True
                grand_fn(sessions_data['lick_properties'], ax=ax, filter_outcomes=outcome, opto_split=opto_split, **kwargs)
    
    plt.subplots_adjust(hspace=3, wspace=3)
    plt.tight_layout()
    
    output_path = os.path.join(save_path, f'psychometric_analysis_{subject}_{data_paths[-1].split("_")[-2]}_{data_paths[0].split("_")[-2]}.pdf')
    fig.savefig(output_path)
    plt.close(fig)
    
    return output_path

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_psychometric_analysis_less_detailed(sessions_data, data_paths, save_path=SAVE_PATH):
    """Generate comprehensive psychometric analysis plots."""
    n_sessions = len(data_paths)
    subject = parse_behavior_file_path(data_paths[0])[0]
    
    # Dynamic figure size
    fig = plt.figure(figsize=(90, (2 + n_sessions) * 5))
    gs = fig.add_gridspec(2 + n_sessions, 12)
    
    # Plot configurations
    plot_configs = [
        ('psychometric', plot_psychometric_curve_less, plot_pooled_psychometric_curve_less, plot_grand_average_psychometric_curve_less),
        ('isi_reaction', plot_isi_reaction_time_less, plot_pooled_isi_reaction_time_less, plot_grand_average_isi_reaction_time_less),
        ('reaction_time', plot_reaction_time_curve_less, plot_pooled_reaction_time_curve_less, plot_grand_average_reaction_time_curve_less)
    ]
    
    # Outcome filters and settings for psychometric functions
    outcomes_psy = [
        ('all', False, 'all'),       # First column: all outcomes, no opto split, all blocks
        ('all', True, 'all'),        # Second column: all outcomes, opto split, all blocks
        ('all', True, 'block1'),     # Third column: all outcomes, opto split, block 1
        ('all', True, 'block2')      # Fourth column: all outcomes, opto split, block 2
    ]
    
    # Outcome filters and settings for other functions
    outcomes_others = [
        ('all', False, 'all'),       # First column: all outcomes, no opto split
        ('all', True, 'all'),        # Second column: all outcomes, opto split
        ('rewarded', True, 'block1'), # Third column: rewarded outcomes, opto split
        ('punished', True, 'block2')  # Fourth column: punished outcomes, opto split
    ]
    
    for i, (plot_type, single_fn, pooled_fn, grand_fn) in enumerate(plot_configs):
        # Select appropriate outcomes list based on plot_type
        outcomes = outcomes_psy if plot_type == 'psychometric' else outcomes_others
        
        # Single session plots
        for j, session_data in enumerate(sessions_data['lick_properties']):
            for k, (outcome, opto_split, block_selection) in enumerate(outcomes):
                if plot_type == 'psychometric' or (plot_type != 'psychometric' and opto_split):
                    ax = fig.add_subplot(gs[2 + j, k + (i * 4)])
                    if plot_type == 'psychometric':
                        single_fn(session_data, ax=ax, filter_outcomes=outcome, opto_split=opto_split, block_selection=block_selection)
                    else:
                        single_fn(session_data, ax=ax, filter_outcomes=outcome, opto_split=opto_split)
        
        # Pooled plots
        for k, (outcome, opto_split, block_selection) in enumerate(outcomes):
            if plot_type == 'psychometric' or (plot_type != 'psychometric' and opto_split):
                ax = fig.add_subplot(gs[0, k + (i * 4)])
                kwargs = {'bin_width': 0.05, 'fit_quadratic': True} if plot_type == 'reaction_time' else {}
                if plot_type == 'psychometric':
                    kwargs['fit_logistic'] = True
                    kwargs['block_selection'] = block_selection
                pooled_fn(sessions_data['lick_properties'], ax=ax, filter_outcomes=outcome, opto_split=opto_split, **kwargs)
        
        # Grand average plots
        for k, (outcome, opto_split, block_selection) in enumerate(outcomes):
            if plot_type == 'psychometric' or (plot_type != 'psychometric' and opto_split):
                ax = fig.add_subplot(gs[1, k + (i * 4)])
                kwargs = {'bin_width': 0.05, 'fit_quadratic': True} if plot_type == 'reaction_time' else {}
                if plot_type == 'psychometric':
                    kwargs['fit_logistic'] = True
                    kwargs['block_selection'] = block_selection
                grand_fn(sessions_data['lick_properties'], ax=ax, filter_outcomes=outcome, opto_split=opto_split, **kwargs)
    
    plt.subplots_adjust(hspace=3, wspace=3)
    plt.tight_layout()
    
    output_path = os.path.join(save_path, f'psychometric_analysis_less_detailed_{subject}_{data_paths[-1].split("_")[-2]}_{data_paths[0].split("_")[-2]}.pdf')
    fig.savefig(output_path)
    plt.close(fig)
    
    return output_path

def plot_psychometric_analysis_halves(sessions_data, data_paths, save_path=SAVE_PATH):
    """Generate comprehensive psychometric analysis plots."""
    n_sessions = len(data_paths)
    subject = parse_behavior_file_path(data_paths[0])[0]
    
    # Dynamic figure size
    fig = plt.figure(figsize=(24, (2 + n_sessions) * 5))
    gs = fig.add_gridspec(2 + n_sessions, 3)
    
    # Plot configurations
    plot_configs = [
        ('psychometric', plot_psychometric_neutral_and_halves, plot_pooled_psychometric_neutral_and_halves, plot_grand_average_psychometric_neutral_and_halves)
    ]
    
    # Outcome filters and settings for psychometric functions
    outcomes = [
        ('all', True, 'neutral'),        # Second column: all outcomes, opto split, all blocks
        ('all', True, 'first_half'),     # Third column: all outcomes, opto split, block 1
        ('all', True, 'second_half')      # Fourth column: all outcomes, opto split, block 2
    ]
    
    for i, (plot_type, single_fn, pooled_fn, grand_fn) in enumerate(plot_configs):
        
        # Only one plot config: psychometric
        # Single session plots
        for j, session_data in enumerate(sessions_data['lick_properties']):
            for k, (outcome, opto_split, block_selection) in enumerate(outcomes):
                ax = fig.add_subplot(gs[2 + j, k])
                single_fn(session_data, ax=ax, filter_outcomes=outcome, opto_split=opto_split, block_selection=block_selection)

        # Pooled plots
        for k, (outcome, opto_split, block_selection) in enumerate(outcomes):
            ax = fig.add_subplot(gs[0, k])
            kwargs = {'fit_logistic': True, 'block_selection': block_selection}
            pooled_fn(sessions_data['lick_properties'], ax=ax, filter_outcomes=outcome, opto_split=opto_split, **kwargs)

        # Grand average plots
        for k, (outcome, opto_split, block_selection) in enumerate(outcomes):
            ax = fig.add_subplot(gs[1, k])
            kwargs = {'fit_logistic': True, 'block_selection': block_selection}
            grand_fn(sessions_data['lick_properties'], ax=ax, filter_outcomes=outcome, opto_split=opto_split, **kwargs)
    
    plt.subplots_adjust(hspace=3, wspace=3)
    plt.tight_layout()
    
    output_path = os.path.join(save_path, f'psychometric_analysis_halves_{subject}_{data_paths[-1].split("_")[-2]}_{data_paths[0].split("_")[-2]}.pdf')
    fig.savefig(output_path)
    plt.close(fig)
    
    return output_path

def plot_psychometric_analysis_opto_seq(sessions_data, data_paths, save_path=SAVE_PATH):
    """Generate comprehensive psychometric analysis plots."""
    n_sessions = len(data_paths)
    subject = parse_behavior_file_path(data_paths[0])[0]
    
    # Dynamic figure size
    fig = plt.figure(figsize=(70, (2 + n_sessions) * 5))
    gs = fig.add_gridspec(2 + n_sessions, 6)
    
    # Plot configurations
    plot_configs = [
        ('psychometric', plot_opto_comparison_curve, plot_pooled_opto_comparison_curve, plot_grand_average_opto_comparison_curve),
        ('reaction_time', plot_opto_comparison_reaction_time_curve, plot_pooled_opto_comparison_reaction_time_curve, plot_grand_average_opto_comparison_reaction_time_curve)
    ]
    
    # Outcome filters and settings
    block_modes = [('short', True), ('long', True), ('all', True)]
    
    for i, (plot_type, single_fn, pooled_fn, grand_fn) in enumerate(plot_configs):
        # Single session plots
        for j, session_data in enumerate(sessions_data['lick_properties']):
            for k, (block_type, opto_split) in enumerate(block_modes):
                ax = fig.add_subplot(gs[2 + j, k + (i * 3)])
                single_fn(session_data, ax=ax, block_mode=block_type)

        # Pooled plots
        for k, (block_type, opto_split) in enumerate(block_modes):
            ax = fig.add_subplot(gs[0, k + (i * 3)])
            kwargs = {'bin_width': 0.05, 'fit_quadratic': True} if plot_type == 'reaction_time' else {}
            if plot_type == 'psychometric' and block_modes == 'all':
                kwargs['fit_logistic'] = True
            pooled_fn(sessions_data['lick_properties'], ax=ax, block_mode=block_type, **kwargs)

        # Grand average plots
        for k, (block_type, opto_split) in enumerate(block_modes):
            ax = fig.add_subplot(gs[1, k + (i * 3)])
            kwargs = {'bin_width': 0.05, 'fit_quadratic': True} if plot_type == 'reaction_time' else {}
            if plot_type == 'psychometric':
                kwargs['fit_logistic'] = True
            grand_fn(sessions_data['lick_properties'], ax=ax, block_mode=block_type, **kwargs)
    
    plt.subplots_adjust(hspace=3, wspace=3)
    plt.tight_layout()
    
    output_path = os.path.join(save_path, f'Opto_seq_psychometric_{subject}_{data_paths[-1].split("_")[-2]}_{data_paths[0].split("_")[-2]}.pdf')
    fig.savefig(output_path)
    plt.close(fig)
    
    return output_path 

def analyze_all_conditions(sessions_data, data_paths, save_path=SAVE_PATH):
    """
    Run all three analysis conditions and create combined plot.
    
    Parameters:
    sessions_data: dict from prepare_session_data function
    
    Returns:
    tuple: (pooled_results, short_results, long_results)
    """
    subject = parse_behavior_file_path(data_paths[0])[0]

    return plot_all_rare_trial_analyses(sessions_data, subject, data_paths, save_path=save_path)

if __name__ == "__main__":
    # sort from the latest sessions data
    data_paths = sorted(DATA_PATHS, key=lambda x: x.split('_')[-2], reverse=False)
    subject = parse_behavior_file_path(data_paths[0])[0]

    # Prepare and plot session data
    sessions_data = prepare_session_data(data_paths)

    # # Generate session plots
    # pdf_filename = generate_session_plots_pdf(data_paths)
    # print(f"Session plots saved to: {pdf_filename}")

    # # Plot outcomes
    # outcome_filename = plot_all_sessions_outcome(sessions_data, data_paths)
    # print(f"Outcome plots saved to: {outcome_filename}")

    # # Plot opto sequence performance
    # outcome_filename = plot_all_sessions_opto_seq(sessions_data, data_paths)
    # print(f"Opto performance plots saved to: {outcome_filename}")

    # # plot opto seq psychometric analysis
    # psychometric_filename = plot_psychometric_analysis_opto_seq(sessions_data, data_paths, save_path=SAVE_PATH)
    # print(f"Opto psychometric analysis saved to: {psychometric_filename}")
    
    # # Plot psychometric analysis
    # psychometric_filename = plot_psychometric_analysis(sessions_data, data_paths)
    # print(f"Psychometric analysis saved to: {psychometric_filename}")

    # # Plot psychometric with less detail
    psychometric_filename = plot_psychometric_analysis_less_detailed(sessions_data, data_paths, save_path=SAVE_PATH)
    print(f"Psychometric analysis saved to: {psychometric_filename}")

    # # plot psychometric halves
    # psychometric_filename = plot_psychometric_analysis_halves(sessions_data, data_paths, save_path=SAVE_PATH)
    # print('Psychometric halves analysis saved to:', psychometric_filename)

    # # Rare trial analysis
    # analyze_all_conditions(sessions_data, data_paths)
    # print('Rare trials analysis saved')

    # # Plot majarity of trials
    # plot_all_majority_trial_analysis(sessions_data, subject, data_paths, save_path=SAVE_PATH)
    # print('Majority trials analysis saved')

    # # Plot majority vs rare trials
    # plot_all_rare_vs_majority_analysis(sessions_data, subject, data_paths, save_path=SAVE_PATH)
    # print('Majority vs rare trials analysis saved')

    # # trial_by_trial_adaptation
    # plot_block_transitions(sessions_data, subject, data_paths, save_path=SAVE_PATH)
    # print('Adaptation analysis saved')

    # # rare trial adaptation analysis
    # plot_rare_trial_performance(sessions_data, subject, data_paths, save_path=SAVE_PATH)
    # print('Rare trial adaptation analysis saved')

    # # reaction time analysis for block transtions
    # plot_reaction_time_transitions(sessions_data, subject, data_paths, save_path=SAVE_PATH)
    # print('Reaction time transitions analysis saved') 

    # # reaction time analysis for rare trials
    # plot_reaction_time_rare_trials(sessions_data, subject, data_paths, save_path=SAVE_PATH)
    # print('Reaction time rare trials analysis saved')

    # # Plot post rare trial analysis (we have 2 epochs for each block)
    # plot_post_rare_trial_analysis(sessions_data, subject, data_paths, save_path=SAVE_PATH)
    # print('Post rare trial analysis saved')

    # # Plot psychometric analysis for epochs
    # plot_psychometric_epochs(sessions_data, subject, data_paths, save_path=SAVE_PATH)
    # print('Psychometric epochs analysis saved')

    # # New version of psychometric analysis for control vs opto block
    # plot_psychometric_control_opto_epochs(sessions_data, subject, data_paths, save_path=SAVE_PATH)

    # # trial dynamics analysis
    # analyze_learning_dynamics(sessions_data, subject, data_paths, save_path=SAVE_PATH)
    # print('Learning dynamics analysis saved')

    # # trial dynamic analysis with fitted exponential
    # analyze_learning_dynamics_fitted(sessions_data, subject, data_paths, save_path=SAVE_PATH, example_session_idx=0)
    # print('Learning dynamics fitted analysis saved')
    
    # # Stimulus specific perfromance
    # plot_performance_by_stimulus(sessions_data, subject, data_paths, save_path=SAVE_PATH) 
    # print('Performance by stimulus analysis saved')

    # # Stimulus specific reaction time
    # plot_rt_by_block_status(sessions_data, subject, data_paths, save_path=SAVE_PATH)
    # print('Reaction time by stimulus analysis saved')

    # # statistical analysis for all conditions
    # plot_summary_stats(sessions_data, subject, data_paths, save_path=SAVE_PATH)
    # print('Statistical summary analysis saved')