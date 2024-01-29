#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from modules import RetrieveResults


# automatical adjustment of contrast.

def adjust_contrast(org_img, lower_percentile=75, upper_percentile=99):
    lower = np.percentile(org_img, lower_percentile)
    upper = np.percentile(org_img, upper_percentile)
    img = np.clip((org_img - lower) * 255 / (upper - lower), 0, 255)
    img = img.astype(np.uint8)
    return img


# convert matrix to RGB image.

def matrix_to_img(
        matrix,
        channel,
        binary
        ):
    img = np.zeros((matrix.shape[0], matrix.shape[1], 3))
    for ch in channel:
        img[:,:,ch] = matrix
    if binary:
        img[img > 1] = 255
    else:
        img = adjust_contrast(img)
    return img


# main function for plot

def plot_fig1(ops):
    
    # read mask from in save_path0 in ops.
    [_, mask] = RetrieveResults.run(ops)
    func_ch    = mask['ch'+str(ops['functional_chan'])]['mean_img']
    func_masks = mask['ch'+str(ops['functional_chan'])]['masks']
    anat_ch    = mask['ch'+str(3-ops['functional_chan'])]['mean_img']
    anat_masks = mask['ch'+str(3-ops['functional_chan'])]['masks']
    ref_ch     = mask['ref']['mean_img']
    ref_masks  = mask['ref']['masks']
    
    # functional channel in green.
    func_ch_img = matrix_to_img(func_ch, [1], False)
    # anatomy channel in red.
    anat_ch_img = matrix_to_img(anat_ch, [0], False)
    # superimpose channel.
    super_img = func_ch_img+anat_ch_img
    # reference image.
    ref_ch_img = adjust_contrast(ref_ch)
    
    # functional masks in green.
    func_masks_img = matrix_to_img(func_masks, [1], True)
    # anatomy masks in red.
    anat_masks_img = matrix_to_img(anat_masks, [0], True)
    # channel shared masks.
    shared_masks_img = func_masks_img + anat_masks_img
    # reference image masks in white.
    ref_masks_img = matrix_to_img(ref_masks, [0,1,2], True)
    
    # plot figs.
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    
    # functional channel mean image.
    axs[0,0].imshow(func_ch_img)
    axs[0,0].set_title('functional channel mean image')
    # anatomy channel mean image.
    axs[0,1].imshow(anat_ch_img)
    axs[0,1].set_title('anatomy channel mean image')
    # superimpose image.
    axs[0,2].imshow(super_img)
    axs[0,2].set_title('channel mean images superimpose')
    # suite2p reference image.
    axs[0,3].imshow(ref_ch_img, cmap='gray')
    axs[0,3].set_title('reference image by suite2p')
    
    # functional channel masks.
    axs[1,0].imshow(func_masks_img)
    axs[1,0].set_title('functional channel masks')
    # anatomy channel masks.
    axs[1,1].imshow(anat_masks_img)
    axs[1,1].set_title('anatomy channel masks')
    # channel shared masks.
    axs[1,2].imshow(shared_masks_img)
    axs[1,2].set_title('channel shared masks')
    # reference image masks.
    axs[1,3].imshow(ref_masks_img)
    axs[1,3].set_title('reference image masks')
    
    # adjust layout
    for i in range(2):
        for j in range(4):
            axs[i,j].tick_params(tick1On=False)
            axs[i,j].spines['left'].set_visible(False)
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['top'].set_visible(False)
            axs[i,j].spines['bottom'].set_visible(False)
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
    fig.suptitle('Mean images and masks by cellpose')
    fig.tight_layout()
    
    # save figure
    fig.savefig(os.path.join(
        ops['save_path0'], 'figures',
        'fig1_mask.pdf'),
        dpi=300)
    plt.close()