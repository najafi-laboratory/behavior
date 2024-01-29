#!/usr/bin/env python3

import os
import json
import numpy as np
from datetime import datetime


# main function for setting the ops.npy for suite2p.

def run(args):
    
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
        
    # set data path and save path specified by command line.
    ops['data_path'] = args.data_path
    ops['save_path0'] = args.save_path0
    ops['functional_chan'] = args.functional_chan
    ops['align_by_chan'] = 3-args.functional_chan
    print('Search data files in {}.'.format(ops['data_path']))
    print('Will save processed data in {}'.format(ops['save_path0']))
    print('Set functional channel to ch'+str(ops['functional_chan']))
    
    # create the path for saving data.
    if not os.path.exists(os.path.join(ops['save_path0'])):
        os.makedirs(os.path.join(ops['save_path0']))
    if not os.path.exists(os.path.join(ops['save_path0'], 'figures')):
        os.makedirs(os.path.join(ops['save_path0'], 'figures'))
        
    # save ops.npy to the path.
    np.save(os.path.join(ops['save_path0'], 'ops.npy'), ops)
    print('Parameters setup completed.')
    
    return ops

