#!/usr/bin/env python3

import os
import h5py
import numpy as np
from datetime import datetime
from suite2p import detection_wrapper


# read the projection images.

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


# save the mask results.

def save_mask(
        ops,
        proj_img,
        results_ch1, results_ch2,
        ):
    # file structure:
    # ops['save_path0'] / mask.h5
    # -- mask
    # ---- ch1
    # ------ mean_img
    # ------ ref_img
    # ------ masks
    # ------ max_img
    # ---- ch2
    # ------ mean_img
    # ------ ref_img
    # ------ masks
    # ------ max_img
    f = h5py.File(os.path.join(
        ops['save_path0'], 'mask.h5'), 'w')
    grp = f.create_group('mask')
    ch1 = grp.create_group('ch1')
    ch2 = grp.create_group('ch2')
    for k in results_ch1.keys():
        ch1[k] = results_ch1[k]
        ch2[k] = results_ch2[k]
    ch1['max_img'] = proj_img['max_ch1']
    ch2['max_img'] = proj_img['max_ch2']
    ch1['mean_img'] = proj_img['mean_ch1']
    ch2['mean_img'] = proj_img['mean_ch2']
    ch1['ref_img'] = proj_img['ref_ch1']
    ch2['ref_img'] = proj_img['ref_ch2']
    f.close()


# run suite2p detection wrapper to get roi stat.

def run_suite2p_roi(
        ops,
        f_reg_ch1,
        f_reg_ch2
        ):
    if ops['functional_chan'] == 1:
        f_reg = f_reg_ch1
    if ops['functional_chan'] == 2:
        f_reg = f_reg_ch2
    _, stat = detection_wrapper(
        f_reg=f_reg,
        ops=ops)
    return stat


# get mask matrix from suite2p stat and organize results.

def stat_to_masks_results(
        ops,
        proj_img,
        stat,
        ):
    masks = np.zeros((ops['Ly'], ops['Lx']))
    for n in range(len(stat)):
        ypix = stat[n]['ypix'][~stat[n]['overlap']]
        xpix = stat[n]['xpix'][~stat[n]['overlap']]
        masks[ypix,xpix] = n+1
    results_ch1 = dict(
        input_img = proj_img['ref_ch1'],
        masks = masks)
    results_ch2 = dict(
        input_img = proj_img['ref_ch2'],
        masks = masks)
    return results_ch1, results_ch2


# main function for ROI detection.

def run(ops, f_reg_ch1, f_reg_ch2):
    print('===============================================')
    print('======== detecting regions of interest ========')
    print('===============================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    proj_img = read_proj_img(ops)

    print('Running suite2p functional ROI detection')
    
    print('Found spatial scale set to {}'.format(ops['spatial_scale']))
    stat_func = run_suite2p_roi(
        ops, f_reg_ch1, f_reg_ch2)

    results_ch1, results_ch2 = stat_to_masks_results(
        ops, proj_img, stat_func)

    # save results.
    save_mask(ops, proj_img, results_ch1, results_ch2)
    print('Mask result saved')
    return stat_func
