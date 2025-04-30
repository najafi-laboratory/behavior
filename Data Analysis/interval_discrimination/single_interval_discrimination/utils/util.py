# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:53:39 2025

@author: timst
"""
def get_figsize_from_pdf_spec(rowspan, colspan, pdf_spec):
    # pdf_spec = config['pdf_spec']
    grid_size = pdf_spec['grid_size']
    fig_size = pdf_spec['fig_size']    
    
    cell_width_in = fig_size[0]
    cell_height_in = fig_size[1] 
    
    nrows = grid_size[0]
    ncols = grid_size[1]
 
    cell_width_in = fig_size[0] / ncols
    cell_height_in = fig_size[1] / nrows
    
    width_in  = colspan * cell_width_in
    height_in = rowspan * cell_height_in
    
    return (width_in, height_in)