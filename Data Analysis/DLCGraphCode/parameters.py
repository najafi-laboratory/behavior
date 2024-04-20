import numpy as np
"""
    use this file to define constants, and adjust values that are constant between all graphs.
"""


# Desired File path
FILE_PATH = '/Users/kushald/code/Najafi Lab/Test/_cam0_run000_20231129_112451DLC_resnet50_initialCalibrationFeb1shuffle1_300000.csv'

# Set frame rate for time calculations. As of April 2024, the cameras record in 60 fps. Do not change unless the
# camera fps is changed.
FRAME_RATE = 60

# This is the mask value. Any probability below this value is not graphed.
MASK_VALUE = 0.90

# Value that decides whether to filter data. If set to true, outliers are filtered.
# only implemented in polygonArea for now.
FILTER = True


def get_color(likelihoods):
    conditions = [
        likelihoods > 0.95,
        (likelihoods > 0.9) & (likelihoods <= 0.95)
    ]
    colors = ['darkred', 'red']
    return np.select(conditions, colors, default='pink')


