#!/usr/bin/env python3

import os
import h5py
import numpy as np
from datetime import datetime
from cellpose import models
from cellpose import io
from cellpose.utils import masks_to_outlines
from suite2p import detection_wrapper
from suite2p.detection import roi_stats


#%% utils


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


# use anatomical to label functional channel masks.

def get_label_from_masks(
        ops,
        restuls_ch1,
        restuls_ch2,
        ):
    # specify functional and anatomical masks.
    if ops['functional_chan'] == 1:
        func_masks = restuls_ch1['masks']
        anat_masks = restuls_ch2['masks']
    if ops['functional_chan'] == 2:
        func_masks = restuls_ch2['masks']
        anat_masks = restuls_ch1['masks']
    # labeling neurons.
    # 0 : only in functional.
    # 1 : in functional and marked in anatomical.
    label = []
    if np.max(func_masks) > 0 and np.max(anat_masks) > 0:
        for i in np.unique(func_masks)[1:]:
            # both channels are available.
            if ops['nchannels'] == 2:
                mul = (func_masks==i) * anat_masks
                # set to 0 if only exist in functional channel.
                # set to 1 if there is overlap between channels.
                label.append(np.sum(mul).astype('bool').astype('int16'))
            # one channel data is dummy.
            if ops['nchannels'] == 1:
                label.append(0)
        label = np.array(label).reshape(-1)
    else:
        raise ValueError('No masks found.')
    return label


# save the mask results.

def save_mask(
        ops,
        proj_img,
        restuls_ch1, restuls_ch2,
        label,
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
    # ---- label
    f = h5py.File(os.path.join(
        ops['save_path0'], 'mask.h5'), 'w')
    grp = f.create_group('mask')
    f['label'] = label
    ch1 = grp.create_group('ch1')
    ch2 = grp.create_group('ch2')
    for k in restuls_ch1.keys():
        ch1[k] = restuls_ch1[k]
        ch2[k] = restuls_ch2[k]
    ch1['max_img'] = proj_img['max_ch1']
    ch2['max_img'] = proj_img['max_ch2']
    ch1['mean_img'] = proj_img['mean_ch1']
    ch2['mean_img'] = proj_img['mean_ch2']
    ch1['ref_img'] = proj_img['ref_ch1']
    ch2['ref_img'] = proj_img['ref_ch2']
    f.close()
    
    
#%% ppc


# identify functional and anatomical channels.

def get_channel_img(
        ops,
        proj_img,
        ch
        ):
    # functional channel.
    if ch == ops['functional_chan']:
        img = proj_img['ref_ch'+str(ops['functional_chan'])]
        return img
    # anatomical channel.
    if ch == 3-ops['functional_chan']:
        img = proj_img['ref_ch'+str(3-ops['functional_chan'])]
        return img

'''
img = proj_img['ref_ch2']
model = models.Cellpose(model_type=ops['pretrained_model'])
masks, flows, styles, diams = model.eval(
    img,
    diameter=12,
    flow_threshold=ops['flow_threshold'],
    channels=[[0,0]],
    channel_axis=0)
from cellpose import plot
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,5))
plot.show_segmentation(fig, img, masks, flows[0], channels=[[0,0]])
plt.tight_layout()
plt.show()
'''

# run cellpose on one image for cell detection and save the results.

def run_cellpose(ops, img):
    # initialize cellpose pretrained model.
    model = models.Cellpose(model_type=ops['pretrained_model'])
    # run cellpose on the given image with the shape of Lx*Ly.
    masks, flows, styles, diams = model.eval(
        img,
        diameter=ops['diameter'] if ops['diameter'] != 0 else None,
        flow_threshold=ops['flow_threshold'],
        channels=[[0,0]],
        channel_axis=0)
    # extract cell outlines.
    outlines = masks_to_outlines(masks)
    return [masks, flows, styles, diams, outlines]


# save cell segmentation to file XXX_seg.npy and reorganize it.

def save_cellpose_results(img, masks, flows, diams, outlines, file_names):
    io.masks_flows_to_seg(img, masks, flows, diams, file_names, [[0,0]])
    restuls = dict(
        input_img = img,
        masks = masks)
    return restuls
    

# identify two channels and get the best results.

def cellpose_eval(
        ops,
        proj_img,
        ch,
        file_names,
        ):
    # functional channel.
    if ch == ops['functional_chan']:
        # find the result with most neurons.
        results = []
        img = [proj_img['ref_ch'+str(ops['functional_chan'])],
               proj_img['max_ch'+str(ops['functional_chan'])]]
        results = [
            run_cellpose(ops, img[0]),
            run_cellpose(ops, img[1])]
        num = [len(np.unique(results[0][0]))-1,
               len(np.unique(results[1][0]))-1]
        masks, flows, styles, diams, outlines = results[np.argmax(num)]
        img = img[np.argmax(num)]
        restuls = save_cellpose_results(
            img, masks, flows, diams, outlines, file_names)
    # anatomical channel.
    if ch == 3-ops['functional_chan']:
        # use reference image.
        img = proj_img['ref_ch'+str(3-ops['functional_chan'])]
        masks, flows, styles, diams, outlines = run_cellpose(ops, img)
        restuls = save_cellpose_results(
            img, masks, flows, diams, outlines, file_names)
    return restuls


# run cellpose on mean channel image and reference image.

def get_mask(
        ops,
        proj_img
        ):
    # ch1 image.
    print('Running cellpose on ch1 image')
    results_ch1 = cellpose_eval(
        ops, proj_img, 1,
        os.path.join(ops['save_path0'], 'temp', 'mask_ch1'))
    # ch2 image.
    print('Running cellpose on ch2 image')
    results_ch2 = cellpose_eval(
        ops, proj_img, 2,
        os.path.join(ops['save_path0'], 'temp', 'mask_ch2'))
    return results_ch1, results_ch2


# reconstruct stat file for suite2p from mask.
# https://github.com/MouseLand/suite2p/issues/292.

def get_stat(
        ops,
        restuls_ch1,
        restuls_ch2,
        ):
    if ops['functional_chan'] == 1:
        masks = restuls_ch1['masks']
    if ops['functional_chan'] == 2:
        masks = restuls_ch2['masks']
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



#%% crbl


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


#%% main

# main function for ROI detection.

def run(ops, f_reg_ch1, f_reg_ch2):
    print('===============================================')
    print('============== detecting neurons ==============')
    print('===============================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    if ops['brain_region'] not in ['ppc', 'crbl']:
        raise ValueError('brain_region can only be ppc or crbl')
        
    proj_img = read_proj_img(ops)
    
    # ppc uses cell detection restuls as ROIs.
    
    if ops['brain_region'] == 'ppc':
        print('Got brain_region = ppc')
        print('Running cellpose detection')
        
        results_ch1, results_ch2 = get_mask(ops, proj_img)
        print('Running cellpose completed')
        
        label = get_label_from_masks(ops, results_ch1, results_ch2)

        stat_func = get_stat(ops, results_ch1, results_ch2)

    # crbl uses suite2p functional ROI detection.
    
    if ops['brain_region'] == 'crbl':
        print('Got brain_region = crbl')
        print('Running suite2p functional ROI detection')
        
        stat_func = run_suite2p_roi(
            ops, f_reg_ch1, f_reg_ch2)
        
        results_ch1, results_ch2 = stat_to_masks_results(
            ops, proj_img, stat_func)
        
        label = get_label_from_masks(ops, results_ch1, results_ch2)
    
    # save results.
    save_mask(ops, proj_img, results_ch1, results_ch2, label)
    print('Mask result saved')
    return stat_func