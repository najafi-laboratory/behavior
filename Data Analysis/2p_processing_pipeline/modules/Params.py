#!/usr/bin/env python3

import os
import json
import numpy as np
from datetime import datetime


# main function for setting the ops.npy for suite2p.
def run():
    print('===============================================')
    print('============ configuring parameters ===========')
    print('===============================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # read the structured config json file.
    with open('./config.json', 'r') as file:
        params = json.load(file)
    # convert it to ops.npy structure by removing the first layer.
    ops = dict()
    for key in params.keys():
        ops.update(params[key])
    # create the path for saving data.
    if not os.path.exists(os.path.join(ops['save_path0'])):
        os.makedirs(os.path.join(ops['save_path0']))
    # save ops.npy to the path.
    np.save(os.path.join(ops['save_path0'], 'ops.npy'), ops)
    print('Parameters setup completed.')
    return ops

