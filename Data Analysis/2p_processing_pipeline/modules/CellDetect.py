#!/usr/bin/env python3

import os
import h5py
import numpy as np
from datetime import datetime
from cellpose import models
from cellpose import io
from cellpose.utils import masks_to_outlines
from suite2p.detection import roi_stats


def read_proj_img(
        ops
        ):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'temp', 'proj_img.h5'),
        'r')
    proj_img = dict()
    for k in f['proj_img'].keys():
        proj_img[k] = np.array(f['proj_img'][k])
    f.close()
    return proj_img


def cellpose_eval(
        mean_img,
        file_names,
        model_type='cyto',
        diameter=None,
        ):
    model = models.Cellpose(model_type=model_type)
    masks, flows, styles, diams = model.eval(
        mean_img,
        diameter=diameter,
        channels=[[0,0]],
        channel_axis=0)
    outlines = masks_to_outlines(masks)
    io.masks_flows_to_seg(mean_img, masks, flows, diams, file_names, [[0,0]])
    restuls = dict(
        mean_img = mean_img,
        masks = masks,
        outlines = outlines,
        diams = diams,
        )
    return restuls


def get_mask(
        ops,
        proj_img
        ):
    restuls_ref = cellpose_eval(
        proj_img['reg_ref'],
        os.path.join(ops['save_path0'], 'temp', 'mask_ref'))
    restuls_ch1 = cellpose_eval(
        proj_img['mean_ch1'],
        os.path.join(ops['save_path0'], 'temp', 'mask_ch1'))
    restuls_ch2 = cellpose_eval(
        proj_img['mean_ch2'],
        os.path.join(ops['save_path0'], 'temp', 'mask_ch2'))
    return restuls_ref, restuls_ch1, restuls_ch2


def get_stat(
        ops,
        masks
        ):
    stat = []
    if np.max(masks) > 0:
        for u_ix, u in enumerate(np.unique(masks)[1:]):
            ypix, xpix = np.nonzero(masks==u)
            npix = len(ypix)
            stat.append(
                {'ypix': ypix,
                 'xpix': xpix,
                 'npix': npix,
                 'lam': np.ones(npix, np.float32),
                 'med': [np.mean(ypix), np.mean(xpix)]})
        stat = np.array(stat)
        stat = roi_stats(stat, ops['Ly'], ops['Lx'])
    return stat


def save_mask(
        ops,
        proj_img,
        restuls_ref, restuls_ch1, restuls_ch2
        ):
    f = h5py.File(os.path.join(
        ops['save_path0'], 'mask.h5'), 'w')
    grp = f.create_group('mask')
    ch1 = grp.create_group('ch1')
    ch2 = grp.create_group('ch2')
    ref = grp.create_group('ref')
    for k in restuls_ref.keys():
        ch1[k] = restuls_ch1[k]
        ch2[k] = restuls_ch2[k]
        ref[k] = restuls_ref[k]
    ch1['max_img'] = proj_img['max_ch1']
    ch2['max_img'] = proj_img['max_ch2']
    f.close()


def run(ops):
    print('===============================================')
    print('============== detecting neurons ==============')
    print('===============================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    proj_img = read_proj_img(ops)
    restuls_ref, restuls_ch1, restuls_ch2 = get_mask(ops, proj_img)
    print('Running cellpose completed')
    save_mask(ops, proj_img, restuls_ref, restuls_ch1, restuls_ch2)
    print('Mask result saved')
    stat_ref = get_stat(ops, restuls_ref['masks'])
    stat_ch1 = get_stat(ops, restuls_ch1['masks'])
    stat_ch2 = get_stat(ops, restuls_ch2['masks'])
    print('Found {} cells in reference image'.format(len(stat_ref)))
    print('Found {} cells in channel 1 mean image'.format(len(stat_ch1)))
    print('Found {} cells in channel 2 mean image'.format(len(stat_ch2)))
    return [stat_ref, stat_ch1, stat_ch2]

