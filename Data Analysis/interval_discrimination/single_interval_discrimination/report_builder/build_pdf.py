# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 15:11:56 2025

@author: timst
"""
import os
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from pdf2image import convert_from_path

def build_pdf_from_registry(config, subjectIdx, registry_path='registry/figure_registry.json', output_path='reports', grid_size=(4, 8)):
    # with open(registry_path, 'r') as f:
    #     registry = json.load(f)

    # subject_id = config['list_config'][subjectIdx]['subject_name']
    # figures = [v for v in registry.values() if v['subject'] == subject_id]
    # figures.sort(key=lambda x: x['figure_id'])  # optional: sort by ID

    # os.makedirs(output_path, exist_ok=True)
    # pdf_path = os.path.join(output_path, f"{subject_id}_report.pdf")
    
    # with PdfPages(pdf_path) as pdf:
    #     for fig_meta in figures:
    #         img = plt.imread(fig_meta['path'])
    #         fig, ax = plt.subplots(figsize=(8, 6))
    #         ax.imshow(img)
    #         ax.axis('off')
    #         ax.set_title(fig_meta['caption'], fontsize=12)
    #         pdf.savefig(fig)
    #         plt.close(fig)

    # print(f"Saved PDF report to {pdf_path}")
    paint_grid = 0

    # Load registry
    with open(registry_path, 'r') as f:
        registry = json.load(f)

    subject = config['list_config'][subjectIdx]['subject_name']
    
    pdf_spec = config['pdf_spec']


    # Group figures by page
    pages = {}
    for fig_id, meta in registry.items():
        if meta.get('subject') != subject:
            continue
        layout = meta.get('layout', {})
        # page = layout.get('page', 0)
        # pages.setdefault(page, []).append((fig_id, meta))
        page_num = layout.get('page', 0)  # <-- key now becomes an int
        pages.setdefault(page_num, []).append((fig_id, meta))        
        # page_key = layout.get("page_key", "default")
        # pages.setdefault(page_key, []).append((fig_id, meta))        

    os.makedirs(output_path, exist_ok=True)
    pdf_filename = os.path.join(output_path, f"{subject}_report.pdf")

    with PdfPages(pdf_filename) as pdf:
        # for page_num in sorted(pages.keys()):
        for page_key in sorted(pages.keys()):
            layout_spec = pdf_spec.get(page_key, {})  
            print(page_key)
            grid_size = layout_spec['grid_size']
            fig_size = layout_spec['fig_size']            
            
            fig = plt.figure(layout='constrained', figsize=fig_size)
            # fig = plt.figure(layout='constrained', figsize=(30, 15))
            # gs = gridspec.GridSpec(*grid_size, figure=fig, hspace=0.5, wspace=0.3)
            gs = gridspec.GridSpec(*grid_size, figure=fig)
# , left=0, right=1, top=1, bottom=0)
           

            if paint_grid:    
                nrows = grid_size[0]
                ncols = grid_size[1]        
                for i in range(nrows):
                    for j in range(ncols):
                        ax = fig.add_subplot(gs[i, j])
                        ax.set_title(f"({i},{j})")
                        ax.set_xticks([]); ax.set_yticks([])
                    ax.set_facecolor((i/nrows, j/ncols, 0.2))

            for fig_id, meta in pages[page_key]:
                layout = meta.get('layout', {})
                row = layout.get('row', 0)
                col = layout.get('col', 0)
                rowspan = layout.get('rowspan', 1)
                colspan = layout.get('colspan', 1)

                # ax = fig.add_subplot(gs[row:row+rowspan, col:col+colspan])
                ax = fig.add_subplot(gs[row:row+rowspan, col:col+colspan])
                # fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)  
                fig.subplots_adjust(left=0, right=0, top=0, bottom=0, wspace=0, hspace=0)  
                
                # img = mpimg.imread(meta['path'])
                
                # Convert PDF page to image (list of PIL images)
                path = meta['path'].replace('\\', '/')
                path = path.replace('//', '/')
                
                print("Checking path:", path)
                print("Exists?", os.path.exists(path))
                                
                images = convert_from_path(path, dpi=300)                
                img = images[0]  # assuming one page                
                
                ax.imshow(img)
                # ax.imshow(img, aspect='auto')  # or aspect='equal', but 'auto' usually fills
                # Keep aspect ratio based on image dimensions
                # ax.imshow(img, aspect='auto')
                
                # Preserve original aspect ratio by setting box_aspect
                # img_aspect = img.shape[0] / img.shape[1]  # height / width
                # ax.set_box_aspect(img_aspect)                
                                
                ax.axis('off')
                # ax.set_title(meta.get('caption', fig_id), fontsize=9)

            pdf.savefig(fig)
            plt.close(fig)

    print(f"PDF saved to: {pdf_filename}")

