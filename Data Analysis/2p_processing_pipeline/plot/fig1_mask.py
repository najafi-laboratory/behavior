#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from modules import RetrieveResults


save_folder = 'FN8_PPC_011724'



    
def plot_fig1():
    [_, _, mask] = RetrieveResults.run(save_folder)
    mean_img_ch1 = mask['ch1']['mean_img']
    mean_img_ch2 = mask['ch2']['mean_img']
    max_img_ch1 = mask['ch1']['max_img']
    max_img_ch2 = mask['ch2']['max_img']

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Mean image and ROI masks')
    fig.tight_layout()
    fig.savefig('./figures/fig1_mask.pdf', dpi=300)
    fig.savefig('./figures/fig1_mask.png', dpi=300)