#!/usr/bin/env python3

import numpy as np
from scipy.ndimage import binary_dilation
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity


# signal ratio between ROI and its surroundings.

def get_roi_sur_ratio(
        mask_roi,
        anat_img
        ):
    # dilation for mask including surrounding.
    mask_roi_surr = binary_dilation(mask_roi, iterations=5)
    # binary operation to get surrounding mask.
    mask_surr = mask_roi_surr != mask_roi
    # compute mean signal and ratio.
    sig_roi  = np.mean(anat_img * mask_roi)
    sig_surr = np.mean(anat_img * mask_surr)
    ratio = sig_roi / (sig_surr + 1e-10)
    return ratio


# spatial correlation between two channels in ROI.

def get_spat_corr(
        mask_roi,
        func_img, anat_img
        ):
    func_roi = func_img * mask_roi
    anat_roi = anat_img * mask_roi
    corr = cosine_similarity(func_roi, anat_roi)
    return corr


# main function to use anatomical to label functional channel masks.

def run(
        ops,
        masks
        ):
    # specify functional and anatomical masks.
    if ops['functional_chan'] == 1:
        func_masks = mask['ch1']['masks']
        func_img   = mask['ch1']['input_img']
        anat_img = mask['ch2']['input_img']
    if ops['functional_chan'] == 2:
        func_masks = mask['ch2']['masks']
        func_img   = mask['ch2']['input_img']
        anat_img = mask['ch1']['input_img']
    # labeling neurons.
    # 0 : only in functional.
    # 1 : in functional and marked in anatomical.
    if np.max(func_masks) > 0:
        surr_ratio = []
        cos_corr = []
        for i in np.unique(func_masks)[1:]:
            # one channel data is dummy so do nothing.
            if ops['nchannels'] == 1:
                surr_ratio.append(0)
                cos_corr.append(0)
            # both channels are available.
            if ops['nchannels'] == 2:
                # get ROI mask.
                mask_roi = (func_masks==i).copy()
                # get signal ratio between roi and its surroundings.
                r = get_roi_sur_ratio(mask_roi, anat_img)
                # get spatial correlation between two channels in ROI.
                c = get_spat_corr(mask_roi, func_img, anat_img)
                # collect results.
                surr_ratio.append(r)
                cos_corr.append(c)
        gmm = GaussianMixture(n_components=2, means_init=([[0.0],[0.5]]))
        label = gmm.fit_predict(np.array(surr_ratio).reshape(-1,1))
    else:
        raise ValueError('No masks found.')
    return label
