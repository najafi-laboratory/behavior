# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 17:04:34 2025

@author: timst
"""
import os
import json

def update_figure_registry(fig_meta, config):
    # print('')
    registry_path = config.get('registry_path', 'registry/figure_registry.json')
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)

    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {}

    registry[fig_meta['figure_id']] = fig_meta

    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
