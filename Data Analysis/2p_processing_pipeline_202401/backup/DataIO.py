#!/usr/bin/env python3

import gc
import os
import math
import tifffile
import numpy as np
from tqdm import tqdm
from datetime import datetime
from suite2p.io.tiff import open_tiff


# read the tif files given the filename list.

def read_tif_to_np(
        ops,
        ch_files
        ):
    # no channel data found.
    if len(ch_files) == 0:
        ch_data = np.zeros((1, ops['Lx'], ops['Ly']), dtype='float32')
    # read into array.
    else:
        ch_data = np.empty((0, ops['Lx'], ops['Ly']), dtype='float32')
        for f in tqdm(ch_files):
            data = tifffile.imread(os.path.join(ops['data_path'], f))
            ch_data = np.concatenate((ch_data, data), axis=0)
            # release memory.
            del data
            gc.collect()
    return ch_data

