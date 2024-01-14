#!/usr/bin/env python3

import os
import json


def run():
    print('===============================================')
    print('============ configuring parameters ===========')
    print('===============================================')
    with open(os.path.join('config', 'params.json'), 'r') as file:
        params = json.load(file)
    ops = dict()
    for key in params.keys():
        ops.update(params[key])
    print('Parameters setup completed')
    return ops

