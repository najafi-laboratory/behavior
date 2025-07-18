import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')

import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from Module.Reader import load_mat
from Module.session_name_parse import parse_behavior_file_path
from Module.Trial_properties import extract_session_properties
from Module.Licking_properties import extract_lick_properties

from plotter.session_structure import plot_trial_outcome_vs_type_with_isi
from plotter.Outcome import plot_all_sessions_with_gridspec
from plotter.licking import (
    plot_psychometric_curve, plot_pooled_psychometric_curve, plot_grand_average_psychometric_curve,
    plot_isi_reaction_time, plot_pooled_isi_reaction_time, plot_grand_average_isi_reaction_time,
    plot_reaction_time_curve, plot_pooled_reaction_time_curve, plot_grand_average_reaction_time_curve
)

# Configuration
# YH24LG ##############################################################################################################################
DATA_PATHS = [
#     'F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_single_interval_discrimination_V_1_11_20250515_181907.mat',
#     'F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_single_interval_discrimination_V_1_11_20250516_155653.mat',
#     'F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_single_interval_discrimination_V_1_11_20250519_175144.mat',
#     'F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_single_interval_discrimination_V_1_11_20250520_184030.mat',
#     'F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_single_interval_discrimination_V_1_11_20250521_192620.mat',
#     'F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_single_interval_discrimination_V_1_11_20250522_181837.mat',
#     'F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_single_interval_discrimination_V_1_11_20250523_173031.mat',
    # 'F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_single_interval_discrimination_V_1_11_20250528_213836.mat',
    # 'F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_single_interval_discrimination_V_1_11_20250529_181219.mat',
    # 'F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_single_interval_discrimination_V_1_11_20250530_215515.mat',
    # 'F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_single_interval_discrimination_V_1_11_20250601_173229.mat',
    # 'F:\Single_Interval_discrimination\Data_behavior\YH24LG\YH24LG_single_interval_discrimination_V_1_11_20250604_191000.mat',
]

SAVE_PATH = 'F:\\Single_Interval_discrimination\\Figures\\YH24LG'
# TS01_LChR ##############################################################################################################################
# DATA_PATHS = [
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250417_075031.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250416_225247.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250414_073145.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250413_111016.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250411_112942.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250410_125310.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250409_190250.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250330_132843.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_8_20250329_113237.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_8_20250328_110455.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250327_184557.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250326_204535.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250324_202107.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250321_195823.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_11_20250521_133332.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_11_20250522_145311.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_11_20250523_135131.mat',
# ]

# Opto-Mid
# DATA_PATHS = [
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_9_20250224_221618.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_9_20250225_191740.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_9_20250226_204526.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_9_20250227_195132.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_9_20250301_213418.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_9_20250302_212005.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_9_20250303_205316.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_9_20250304_195615.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_9_20250305_163852.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_9_20250306_202610.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_9_20250308_214653.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_9_20250309_220849.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250311_195050.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250312_215826.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250314_205555.mat',        
# ]

# Opto-Early
# DATA_PATHS = [
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250316_213144.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250317_213448.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250318_184345.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250320_203341.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250321_195823.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250324_202107.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250326_204535.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250327_184557.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250330_132843.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250331_141036.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250401_154509.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250403_202240.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250404_231556.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250405_113729.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250406_165117.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS01_LChR\\LCHR_TS01_single_interval_discrimination_V_1_10_20250407_100841.mat',        
# ]

# SAVE_PATH = 'F:\\Single_Interval_discrimination\\Figures\\TS01_LChR'

# TS02_LChR ##############################################################################################################################
# DATA_PATHS = [
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250417_085055.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250416_220127.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250415_145638.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250414_172542.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250411_103435.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250410_074411.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250409_201614.mat',
# ]

#Opto Left
# DATA_PATHS = [
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_9_20250225_201421.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_9_20250301_223020.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_9_20250303_215451.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_9_20250305_175626.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_9_20250308_230024.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_9_20250309_230705.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250316_220701.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250317_223308.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250318_175930.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250320_213030.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250327_174456.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250331_150124.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250401_163804.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250403_210843.mat',
# ] 

#Opto Right
# DATA_PATHS = [
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_9_20250224_231949.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_9_20250227_205609.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_9_20250302_222506.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_9_20250304_205131.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_9_20250306_212416.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250324_211801.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250405_124545.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250407_191414.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250409_201614.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250410_074411.mat',
    # 0.5_2.0
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250411_103435.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250415_145638.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250416_220127.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS02_LChR\\LCHR_TS02_single_interval_discrimination_V_1_10_20250417_085055.mat',
# ] 

# SAVE_PATH = 'F:\\Single_Interval_discrimination\\Figures\\TS02_LChR'

# TS06_SChR ##############################################################################################################################
# DATA_PATHS = [
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250416_205942.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250415_091444.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250414_093242.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250411_092808.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250410_181115.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250409_091345.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250408_081805.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_11_20250520_171227.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_11_20250521_162255.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_11_20250522_150940.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_11_20250523_142618.mat',
# ]

# Early Sessions
# DATA_PATHS = [
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250202_211817.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250203_194230.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250204_214022.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250205_203111.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250206_194910.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250207_203520.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250208_204546.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250209_201603.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250210_204947.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250211_200103.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250212_192216.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250213_200211.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250214_192650.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250215_204923.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250216_200125.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250217_200351.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250219_214316.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250220_193735.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250221_202845.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250222_202429.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250223_214929.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250224_202126.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250225_180751.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250226_193449.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250227_181143.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250228_184705.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250301_195035.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250302_200603.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250303_181215.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250304_181850.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250305_144045.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250306_173937.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250307_182322.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250308_190009.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250309_194452.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250311_172619.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250312_185158.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250314_175841.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250315_193142.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250316_191706.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250317_192051.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250318_201115.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250319_193603.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250321_180418.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250324_184805.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_8_20250325_212742.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250329_122935.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250330_113915.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250401_121236.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250403_080400.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250404_215615.mat',
# ]

# Opto_frontal
# DATA_PATHS = [
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250404_215615.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250405_170954.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250406_150431.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250407_064206.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250408_081805.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250409_091345.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250410_181115.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250411_092808.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250415_091444.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS06_SChR\\SCHR_TS06_single_interval_discrimination_V_1_10_20250416_205942.mat',
# ]
# SAVE_PATH = 'F:\\Single_Interval_discrimination\\Figures\\TS06_SChR'

# TS07_SChR ##############################################################################################################################
# DATA_PATHS = [
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_10_20250416_200308.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_10_20250415_133410.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_10_20250414_111734.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_10_20250413_101621.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_10_20250411_125506.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_10_20250410_191332.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_10_20250409_100600.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_10_20250408_091223.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_11_20250519_143224.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_11_20250520_113325.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_11_20250521_153605.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_11_20250522_154441.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_11_20250523_145112.mat',
# ]

# EArly Sessions
# DATA_PATHS = [
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_8_20250202_213027.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_8_20250203_195348.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_8_20250204_215113.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_8_20250205_205733.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_8_20250206_201914.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_8_20250207_211831.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_8_20250208_211540.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_8_20250209_203634.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS07_SChR\\SCHR_TS07_single_interval_discrimination_V_1_8_20250210_210146.mat',
# ]

# SAVE_PATH = 'F:\\Single_Interval_discrimination\\Figures\\TS07_SChR'

# TS08_SChR ##############################################################################################################################
# DATA_PATHS = [
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250416_181214.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250415_072926.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250414_144258.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250411_080113.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250410_083251.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250409_072225.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250408_101149.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_11_20250520_145038.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_11_20250521_173839.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_11_20250522_161149.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_11_20250523_161505.mat',
# ]

# Early Sessions
# DATA_PATHS = [
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_8_20250219_140438.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_8_20250220_171938.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_8_20250221_163247.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_8_20250222_174259.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_8_20250223_193305.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_8_20250224_191424.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_8_20250225_140656.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_8_20250226_172556.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_8_20250227_102122.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_8_20250228_120127.mat',
# ]

# DATA_PATHS = [
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250327_211331.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250328_140930.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250329_103723.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250330_095530.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250331_112110.mat',
    # 'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250401_133940.mat',
    # interval 0.5_2.0
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250404_190523.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250405_193322.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250406_125720.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250407_081958.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250408_101149.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250409_072225.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250410_083251.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250411_080113.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250414_144258.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250415_072926.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS08_SChR\\SCHR_TS08_single_interval_discrimination_V_1_10_20250416_181214.mat',
# ]

# SAVE_PATH = 'F:\\Single_Interval_discrimination\\Figures\\TS08_SChR'

# TS09_SChR ##############################################################################################################################
# DATA_PATHS = [
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250416_190712.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250415_081728.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250414_153403.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250411_083631.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250410_092709.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250409_081814.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250408_110157.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_11_20250519_153025.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_11_20250520_141023.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_11_20250521_163618.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_11_20250522_163802.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_11_20250523_154646.mat',
# ]
# Early Sessions
# DATA_PATHS = [
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_8_20250219_142634.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_8_20250220_112515.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_8_20250221_164603.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_8_20250222_175431.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_8_20250223_194106.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_8_20250224_191907.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_8_20250225_140615.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_8_20250226_172646.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_8_20250227_113639.mat',
# ]

# Opto_PPC
# DATA_PATHS = [
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250327_220834.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250328_145956.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250329_112423.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250330_104559.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250331_121221.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250401_144409.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250403_192913.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250404_203608.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250405_202305.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250406_141128.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250407_091521.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250409_081814.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250410_092709.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250411_083631.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250414_153403.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250415_081728.mat',
#     'F:\\Single_Interval_discrimination\\Data_behavior\\TS09_SChR\\SCHR_TS09_single_interval_discrimination_V_1_10_20250416_190712.mat',
# ]

# SAVE_PATH = 'F:\\Single_Interval_discrimination\\Figures\\TS09_SChR'

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
        'lick_properties': []
    }
    
    for path in data_paths:
        data = load_session_data(path)
        sessions_data['dates'].append(data['date'])
        sessions_data['outcomes'].append(data['trial_properties']['outcome'])
        sessions_data['opto_tags'].append(data['trial_properties']['opto_tag'])
        sessions_data['trial_types'].append(data['trial_properties']['trial_type'])
        sessions_data['lick_properties'].append(data['lick_properties'])
    
    return sessions_data

def plot_all_sessions_outcome(sessions_data, data_paths, save_path=SAVE_PATH):
    """Plot all sessions' outcomes."""
    subject = parse_behavior_file_path(data_paths[0])[0]
    fig = plot_all_sessions_with_gridspec(
        sessions_data['outcomes'], sessions_data['dates'],
        sessions_data['trial_types'], sessions_data['opto_tags'],
        figsize=(40, 20)
    )
    
    name = f'outcome_{subject}_{data_paths[-1].split("_")[-2]}_{data_paths[0].split("_")[-2]}.pdf'
    output_path = os.path.join(save_path, name)
    fig.savefig(output_path)
    plt.close(fig)
    
    return name

def plot_psychometric_analysis(sessions_data, data_paths, save_path=SAVE_PATH):
    """Generate comprehensive psychometric analysis plots."""
    n_sessions = len(data_paths)
    subject = parse_behavior_file_path(data_paths[0])[0]
    
    # Dynamic figure size
    fig = plt.figure(figsize=(70, (2 + n_sessions) * 5))
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
    
    plt.subplots_adjust(hspace=2, wspace=2)
    plt.tight_layout()
    
    output_path = os.path.join(save_path, f'psychometric_analysis_{subject}_{data_paths[-1].split("_")[-2]}_{data_paths[0].split("_")[-2]}.pdf')
    fig.savefig(output_path)
    plt.close(fig)
    
    return output_path

if __name__ == "__main__":
    # Generate session plots
    pdf_filename = generate_session_plots_pdf(DATA_PATHS)
    print(f"Session plots saved to: {pdf_filename}")
    
    # # # Prepare and plot session data
    # sessions_data = prepare_session_data(DATA_PATHS)
    
    # # # Plot outcomes
    # outcome_filename = plot_all_sessions_outcome(sessions_data, DATA_PATHS)
    # print(f"Outcome plots saved to: {outcome_filename}")
    
    # # Plot psychometric analysis
    # psychometric_filename = plot_psychometric_analysis(sessions_data, DATA_PATHS)
    # print(f"Psychometric analysis saved to: {psychometric_filename}")