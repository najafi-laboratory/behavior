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
    if args.brain_region not in ['ppc', 'crbl']:
        raise ValueError('brain_region can only be ppc or crbl')
    elif args.brain_region == 'ppc':
        with open('./config_ppc.json', 'r') as file:
            params = json.load(file)
    elif args.brain_region == 'crbl':
        with open('./config_crbl.json', 'r') as file:
            params = json.load(file)
            
    # convert to ops.npy for suite2p by removing the first layer.
    ops = dict()
    for key in params.keys():
        ops.update(params[key])

    # set params specified by command line.
    ops['Lx']              = 512
    ops['Ly']              = 512
    ops['spatial_scale']   = args.spatial_scale
    ops['data_path']       = args.data_path
    ops['save_path0']      = args.save_path0
    ops['nchannels']       = args.nchannels
    ops['functional_chan'] = args.functional_chan
    ops['align_by_chan']   = 3-args.functional_chan
    ops['brain_region']    = args.brain_region
    print('Search data files in {}'.format(ops['data_path']))
    print('Will save processed data in {}'.format(ops['save_path0']))
    print('Processing {} channel data'.format(ops['nchannels']))
    print('Set functional channel to ch'+str(ops['functional_chan']))

    # create the path for saving data.
    if not os.path.exists(os.path.join(ops['save_path0'])):
        os.makedirs(os.path.join(ops['save_path0']))
    if not os.path.exists(os.path.join(ops['save_path0'], 'temp')):
        os.makedirs(os.path.join(ops['save_path0'], 'temp'))
    if not os.path.exists(os.path.join(ops['save_path0'], 'figures')):
        os.makedirs(os.path.join(ops['save_path0'], 'figures'))

    # save ops.npy to the path.
    np.save(os.path.join(ops['save_path0'], 'ops.npy'), ops)
    print('Parameters setup for {} completed'.format(ops['brain_region']))

    return ops
