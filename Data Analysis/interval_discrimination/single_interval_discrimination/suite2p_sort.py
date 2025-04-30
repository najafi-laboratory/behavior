# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 10:50:40 2025

@author: timst
"""

import numpy as np
import os

# Path to suite2p output folder (e.g., plane0)

output_path = 'D:/git/2p_imaging/passive_interval_oddball_202412/results/TS03/YH24LG_CRBL_crux1_20250425_2afc/suite2p/plane0'

# Load stat and iscell arrays
stat = np.load(os.path.join(output_path, 'stat.npy'), allow_pickle=True)
n_rois = len(stat)

# Set all to iscell
iscell = np.ones((n_rois, 2), dtype=np.float32)

# Mark all as cells
iscell[:, 0] = 1  # 1 = cell, 0 = not cell
iscell[:, 1] = 0.9  # probability/confidence of being a cell (can be any value between 0 and 1)

# Mark all as not cells
# iscell[:, 0] = 0  
# np.save(os.path.join(output_path, 'iscell.npy'), iscell)

# Save back
np.save(os.path.join(output_path, 'iscell.npy'), iscell)