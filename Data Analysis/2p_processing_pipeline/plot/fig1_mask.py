#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from modules import RetrieveResults


# automatical adjustment of contrast.
def adjust_contrast(org_img, lower_percentile=25, upper_percentile=99):
    lower = np.percentile(org_img, lower_percentile)
    upper = np.percentile(org_img, upper_percentile)
    img = np.clip((org_img - lower) * 255 / (upper - lower), 0, 255)
    img = img.astype(np.uint8)
    return img

    
def plot_fig1(ops):
    # read mask from in save_path0 in ops.
    [_, mask] = RetrieveResults.run(ops)
    mean_img_ch1 = mask['ch1']['mean_img']
    mean_img_ch2 = mask['ch2']['mean_img']
    mean_img_ref = mask['ref']['mean_img']
    outlines_ch1 = mask['ch1']['outlines']
    outlines_ch2 = mask['ch2']['outlines']
    outlines_ref = mask['ref']['outlines']
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    # channel 1 mean image.
    axs[0,0].imshow(adjust_contrast(mean_img_ch1), cmap='gray')
    axs[0,0].set_title('channel 1 mean image')
    # channel 1 outlines superimposing on mean image.
    outline_img_ch1 = np.zeros((outlines_ch1.shape[0], outlines_ch1.shape[1], 4))
    outline_img_ch1[outlines_ch1, :] = [255, 0, 0, 255]
    axs[1,0].imshow(adjust_contrast(mean_img_ch1), cmap='gray')
    axs[1,0].imshow(outline_img_ch1)
    axs[1,0].set_title('channel 1 mask')
    # channel 2 mean image.
    axs[0,1].imshow(adjust_contrast(mean_img_ch2), cmap='gray')
    axs[0,1].set_title('channel 2 mean image')
    # channel 2 outlines superimposing on mean image.
    outline_img_ch2 = np.zeros((outlines_ch2.shape[0], outlines_ch2.shape[1], 4))
    outline_img_ch2[outlines_ch2, :] = [255, 0, 0, 255]
    axs[1,1].imshow(adjust_contrast(mean_img_ch2), cmap='gray')
    axs[1,1].imshow(outline_img_ch2)
    axs[1,1].set_title('channel 2 mask')
    # suite2p reference frame
    axs[0,2].imshow(adjust_contrast(mean_img_ref), cmap='gray')
    axs[0,2].set_title('suite2p reference image')
    # channel 2 outlines superimposing on mean image.
    outline_img_ref = np.zeros((outlines_ref.shape[0], outlines_ref.shape[1], 4))
    outline_img_ref[outlines_ref, :] = [255, 0, 0, 255]
    axs[1,2].imshow(adjust_contrast(mean_img_ref), cmap='gray')
    axs[1,2].imshow(outline_img_ref)
    axs[1,2].set_title('reference image mask')
    # adjust layout
    for i in range(2):
        for j in range(3):
            axs[i,j].tick_params(tick1On=False)
            axs[i,j].spines['left'].set_visible(False)
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['top'].set_visible(False)
            axs[i,j].spines['bottom'].set_visible(False)
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
    fig.suptitle('ROI and masks')
    fig.tight_layout()
    # save figure
    fig.savefig(os.path.join(ops['save_path0'], 'fig1_mask.png'), dpi=300)
    plt.close()