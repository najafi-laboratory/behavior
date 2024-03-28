#!/usr/bin/env python3

import os
import h5py
import numpy as np
from skimage.measure import label


'''

# read saved ops.npy given a folder in ./results.

def read_ops(save_folder):
    ops = np.load(
        os.path.join('./results', save_folder, 'ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join('./results', save_folder)
    return ops

ops = read_ops('C1')
labeled_image = label(masks, connectivity=1)
'''

# read raw results from suite2p pipeline.

def read_raw(ops):
    F    = np.load(os.path.join(ops['save_path'], 'F.npy'),    allow_pickle=True)
    Fneu = np.load(os.path.join(ops['save_path'], 'Fneu.npy'), allow_pickle=True)
    stat = np.load(os.path.join(ops['save_path'], 'stat.npy'), allow_pickle=True)
    return [F, Fneu, stat]


# get metrics for ROIs.

def get_metrics(ops, stat):
    # rearrange existing statistics for masks.
    # https://suite2p.readthedocs.io/en/latest/outputs.html#stat-npy-fields
    footprint = np.array([stat[i]['footprint']    for i in range(len(stat))])
    skew      = np.array([stat[i]['skew']         for i in range(len(stat))])
    # compute connetivity of ROIs.
    masks = stat_to_masks(ops, stat)
    connect = []
    for i in np.unique(masks)[1:]:
        # find a mask with one roi.
        m = masks.copy() * (masks == i)
        # find component number.
        connect.append(np.max(label(m, connectivity=1)))
    connect = np.array(connect)
    return footprint, skew, connect


# threshold the statistics to keep good ROIs.

def thres_stat(ops, stat):
    footprint, skew, connect = get_metrics(ops, stat)
    # find bad roi indice.
    bad_roi_id = set()
    bad_roi_id = bad_roi_id.union(
        bad_roi_id,
        np.where((footprint==0) | (footprint==3))[0])
    bad_roi_id = bad_roi_id.union(
        bad_roi_id,
        np.where((skew<np.median(skew)-0.5) | (skew>np.median(skew)+5))[0])
    bad_roi_id = bad_roi_id.union(
        bad_roi_id,
        np.where(connect>3)[0])
    # convert set to numpy array for indexing.
    bad_roi_id = np.array(list(bad_roi_id))
    return bad_roi_id


# reset bad ROIs in the masks to nothing.

def reset_roi(
        bad_roi_id,
        F, Fneu, stat
        ):
    # reset bad roi.
    for i in bad_roi_id:
        stat[i]   = None
        F[i,:]    = 0
        Fneu[i,:] = 0
    # find good roi indice.
    good_roi_id = np.where(np.sum(F, axis=1)!=0)[0]
    # keep good roi signals.
    fluo = F[good_roi_id,:]
    neuropil = Fneu[good_roi_id,:]
    stat = stat[good_roi_id]
    return fluo, neuropil, stat
    

# save results into npy files.

def save_qc_results(
        ops,
        fluo, neuropil, stat
        ):
    np.save(os.path.join(ops['save_path0'], 'fluo.npy'), fluo)
    np.save(os.path.join(ops['save_path0'], 'neuropil.npy'), neuropil)
    np.save(os.path.join(ops['save_path0'], 'stat.npy'), stat)
    np.save(os.path.join(ops['save_path0'], 'ops.npy'), ops)


# convert stat.npy results to ROI masks matrix.

def stat_to_masks(ops, stat):
    masks = np.zeros((ops['Ly'], ops['Lx']))
    for n in range(len(stat)):
        ypix = stat[n]['ypix']
        xpix = stat[n]['xpix']
        masks[ypix,xpix] = n+1
    return masks


# read raw_voltages.h5

def read_raw_voltages(ops):
    try:
        f = h5py.File(
            os.path.join(ops['save_path0'], 'raw_voltages.h5'),
            'r')
        raw_voltages = dict()
        for k in f['raw'].keys():
            raw_voltages[k] = np.array(f['raw'][k])
        f.close()
        return raw_voltages
    except:
        print('Fail to read voltage data')
        return None


# main function for quality control

def run(ops):
    [F, Fneu, stat] = read_raw(ops)
    bad_roi_id = thres_stat(ops, stat)
    fluo, neuropil, stat = reset_roi(bad_roi_id, F, Fneu, stat)
    save_qc_results(ops, fluo, neuropil, stat)
    masks = stat_to_masks(ops, stat)
    raw_voltages = read_raw_voltages(ops)
    return [fluo, neuropil, masks, raw_voltages]