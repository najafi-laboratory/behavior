#!/usr/bin/env python3

import os
import json
import numpy as np
from datetime import datetime


def run():
    print('===============================================')
    print('============ configuring parameters ===========')
    print('===============================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join('config', 'params.json'), 'r') as file:
        params = json.load(file)
    ops = dict()
    for key in params.keys():
        ops.update(params[key])
    if not os.path.exists(os.path.join(ops['save_path0'])):
        os.makedirs(os.path.join(ops['save_path0']))
    np.save(os.path.join(ops['save_path0'], 'ops.npy'), ops)
    print('Parameters setup completed')
    return ops

