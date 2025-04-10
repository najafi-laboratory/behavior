# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 15:11:56 2025

@author: timst
"""

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import json

def build_pdf_from_registry(subject_id, registry_path='registry/figure_registry.json', output_path='reports'):
    with open(registry_path, 'r') as f:
        registry = json.load(f)

    figures = [v for v in registry.values() if v['subject'] == subject_id]
    figures.sort(key=lambda x: x['figure_id'])  # optional: sort by ID

    os.makedirs(output_path, exist_ok=True)
    pdf_path = os.path.join(output_path, f"{subject_id}_report.pdf")
    
    with PdfPages(pdf_path) as pdf:
        for fig_meta in figures:
            img = plt.imread(fig_meta['path'])
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(fig_meta['caption'], fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved PDF report to {pdf_path}")

