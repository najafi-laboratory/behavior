#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from postprocess import QualControlDataIO


# automatical adjustment of contrast.

def adjust_contrast(org_img, lower_percentile=50, upper_percentile=99):
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
        img[img >= 1] = 255
    else:
        img = adjust_contrast(img)
    img = img.astype('uint8')
    return img


# label images with yellow and green.

def get_labeled_masks_img(
        func_masks,
        label,
        ):
    # get green image.
    labeled_masks_img = matrix_to_img(func_masks, [1], True)
    # find marked neurons.
    neuron_idx = np.where(label == 1)[0] + 1
    for i in neuron_idx:
        neuron_mask = ((func_masks == i) * 255).astype('uint8')
        labeled_masks_img[:,:,0] += neuron_mask
    return labeled_masks_img


# main function for plot.

def plot_fig1(ops):

    try:
        print('plotting fig1 masks')
        
        # read mask from in save_path0 in ops.
        [_, _, masks, _] = QualControlDataIO.run(ops)

        # 1 channel data.
        if ops['nchannels'] == 1:
            
            # plot figs.
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
            # reference image same as mean.
            axs[0].matshow(adjust_contrast(ops['refImg']), cmap='grey')
            axs[0].set_title('reference image')
    
            # max projection.
            axs[1].matshow(adjust_contrast(ops['max_proj']), cmap='grey')
            axs[1].set_title('max projection')
            
            # ROI masks.
            colors = plt.cm.nipy_spectral(np.linspace(0, 1, int(np.max(masks)+1)))
            np.random.shuffle(colors)
            colors[0,:] = [0,0,0,1]
            cmap = ListedColormap(colors)
            axs[2].matshow(masks, cmap=cmap)
            axs[2].set_title('ROI masks')
            
            # adjust layout
            for i in range(axs.shape[0]):
                axs[i].tick_params(tick1On=False)
                axs[i].spines['left'].set_visible(False)
                axs[i].spines['right'].set_visible(False)
                axs[i].spines['top'].set_visible(False)
                axs[i].spines['bottom'].set_visible(False)
                axs[i].set_xticks([])
                axs[i].set_yticks([])
            fig.suptitle('Channel images and masks by suite2p')
            fig.tight_layout()
            
        # 2 channel data.
        if ops['nchannels'] == 2:
    
            # plot figs.
            fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    
            # functional channel mean image.
            axs[0,0].imshow(func_ch_img)
            axs[0,0].set_title('functional channel')
            # anatomy channel mean image.
            axs[0,1].imshow(anat_ch_img)
            axs[0,1].set_title('anatomy channel')
            # superimpose image.
            axs[0,2].imshow(super_img)
            axs[0,2].set_title('channel images superimpose')
    
            # functional channel masks.
            axs[1,0].imshow(func_masks_img)
            axs[1,0].set_title('functional channel masks')
            # anatomy channel masks.
            axs[1,1].imshow(anat_masks_img)
            axs[1,1].set_title('anatomy channel masks')
            # channel shared masks.
            #axs[1,2].imshow(super_masks)
            axs[1,2].set_title('channel masks superimpose')
    
            # adjust layout
            for i in range(axs.shape[0]):
                for j in range(axs.shape[1]):
                    axs[i,j].tick_params(tick1On=False)
                    axs[i,j].spines['left'].set_visible(False)
                    axs[i,j].spines['right'].set_visible(False)
                    axs[i,j].spines['top'].set_visible(False)
                    axs[i,j].spines['bottom'].set_visible(False)
                    axs[i,j].set_xticks([])
                    axs[i,j].set_yticks([])
            fig.suptitle('Channel images and masks by cellpose')
            fig.tight_layout()
    
        # save figure
        fig.savefig(os.path.join(
            ops['save_path0'], 'figures',
            'fig1_mask.pdf'),
            dpi=300)
        plt.close()
        
    except:
        print('plotting fig1 failed')