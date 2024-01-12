import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models
from cellpose import plot
from cellpose import io
from suite2p.detection import roi_stats


def cellpose_eval(
        img,
        file_names,
        model_type='cyto',
        diameter=None,
        plot_cellpose=True,
        ):
    model = models.Cellpose(model_type=model_type)
    masks, flows, styles, diams = model.eval(
        img,
        diameter=diameter,
        channels=[[0,0]],
        channel_axis=0)
    if plot_cellpose:
        fig = plt.figure(figsize=(15,5))
        plot.show_segmentation(fig, img, masks, flows[0], channels=[[0,0]])
        plt.tight_layout()
        plt.show()
    io.masks_flows_to_seg(img, masks, flows, diams, file_names, [[0,0]])
    return masks


def get_mask(ops, reg_ref, mean_ch1, mean_ch2):
    masks_ref = cellpose_eval(
        reg_ref,
        os.path.join(ops['save_path0'], 'mask_reg_ref'))
    masks_r_ch = cellpose_eval(
        mean_ch1,
        os.path.join(ops['save_path0'], 'mask_r_ch'))
    masks_g_ch = cellpose_eval(
        mean_ch2,
        os.path.join(ops['save_path0'], 'mask_g_ch'))
    return masks_ref, masks_r_ch, masks_g_ch


def get_stat(ops, masks):
    stat = []
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


def run(ops, reg_ref, mean_ch1, mean_ch2):
    print('===============================================')
    print('============== detecting neurons ==============')
    print('===============================================')
    masks_ref, masks_r_ch, masks_g_ch = get_mask(
        ops, reg_ref, mean_ch1, mean_ch2)
    print('Running cellpose completed')
    stat_ref = get_stat(ops, masks_ref)
    stat_r_ch = get_stat(ops, masks_r_ch)
    stat_g_ch = get_stat(ops, masks_g_ch)
    print('Found {} cells in reference image'.format(len(stat_ref)))
    print('Found {} cells in red channel mean image'.format(len(stat_r_ch)))
    print('Found {} cells in green channel mean image'.format(len(stat_g_ch)))
    return [stat_ref, stat_r_ch, stat_g_ch]



