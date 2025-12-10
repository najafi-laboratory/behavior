#!/usr/bin/env python3
"""
QC Quick Check (standalone)

Loads precomputed QC artifacts and emits:
  - roi_overlay.png: all ROI outlines on meanImg
  - roi_gallery.png: grid of sample ROIs with core/edge/ring overlays
  - flag_bars.png: percent flagged per criterion
  - geometry_hist.png: histograms of area & components (if metrics available)

Hardcode your session paths below and run:
    python qc_quickcheck.py
"""

import os, sys, math, json, numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours

# -------------------- HARD-CODED PATHS --------------------
SUITE2P_PLANE0 = "D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/suite2p/plane0"
OUTPUT_PATH     = "D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494"
QC_RESULTS      = os.path.join(OUTPUT_PATH, "qc_results")
OUTDIR          = os.path.join(QC_RESULTS, "quickcheck")

# -------------------- HELPERS --------------------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _load_npy(path, allow_pickle=False, key=None):
    if not os.path.exists(path):
        return None
    arr = np.load(path, allow_pickle=allow_pickle)
    if key is not None and arr.dtype == object:
        try:
            return arr.item().get(key, None)
        except Exception:
            pass
    return arr

def _outline_image_from_labels(labels):
    """Return list of contour arrays for each nonzero label."""
    if labels is None:
        return []
    u = np.unique(labels)
    u = u[u > 0]
    contours_by_id = {}
    for rid in u:
        # find_contours expects floats; threshold 0.5 separates label=rid from others
        mask = (labels == rid).astype(float)
        cs = find_contours(mask, 0.5)
        contours_by_id[rid] = cs
    return contours_by_id

def _plot_overlay(mean_img, labels, save_path, title="ROI overlay", alpha_img=0.9):
    plt.figure(figsize=(8, 8))
    if mean_img is None:
        mean_img = np.zeros(labels.shape, dtype=float)
    # auto-scale contrast
    if mean_img is not None:
        disp_img = mean_img.astype(float)
        disp_img -= disp_img.min()
        disp_img /= disp_img.max() + 1e-9
        # plt.imshow(disp_img, cmap="gray")
    vmin, vmax = np.percentile(mean_img, [2, 98]) if mean_img.size else (0, 1)
    # plt.imshow(mean_img, cmap="gray", vmin=vmin, vmax=vmax, alpha=alpha_img)
    plt.imshow(disp_img, cmap="gray", vmin=vmin, vmax=vmax, alpha=alpha_img)
    contours = _outline_image_from_labels(labels)
    for rid, cs in contours.items():
        for c in cs:
            plt.plot(c[:, 1], c[:, 0], linewidth=0.7)
    n = int((labels > 0).sum())
    plt.title(f"{title}  (n={len(contours)})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def _draw_roi_panel(ax, roi_id, labels, core_xy=None, edge_xy=None, ring_xy=None, mean_img=None, pad=6):
    """Zoomed panel around an ROI label id, overlay core/edge/ring if provided."""
    mask = (labels == roi_id)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        ax.axis("off"); return
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(0, y0 - pad); y1 = min(labels.shape[0]-1, y1 + pad)
    x0 = max(0, x0 - pad); x1 = min(labels.shape[1]-1, x1 + pad)

    if mean_img is None:
        sub_img = np.zeros((y1 - y0 + 1, x1 - x0 + 1))
        vmin, vmax = 0, 1
    else:
        sub_img = mean_img[y0:y1+1, x0:x1+1]
        vmin, vmax = np.percentile(sub_img, [2, 98])

    ax.imshow(sub_img, cmap="gray", vmin=vmin, vmax=vmax)
    # outline
    sub = mask[y0:y1+1, x0:x1+1].astype(float)
    cs = find_contours(sub, 0.5)
    for c in cs:
        ax.plot(c[:, 1], c[:, 0], linewidth=1.0)

    # overlays
    if core_xy is not None:
        y, x = core_xy
        m = (y >= y0) & (y <= y1) & (x >= x0) & (x <= x1)
        ax.scatter(x[m] - x0, y[m] - y0, s=2)
    if edge_xy is not None:
        y, x = edge_xy
        m = (y >= y0) & (y <= y1) & (x >= x0) & (x <= x1)
        ax.scatter(x[m] - x0, y[m] - y0, s=2)
    if ring_xy is not None:
        y, x = ring_xy
        m = (y >= y0) & (y <= y1) & (x >= x0) & (x <= x1)
        ax.scatter(x[m] - x0, y[m] - y0, s=1)

    ax.set_title(f"ROI {roi_id}")
    ax.axis("off")

def _safe_object_array(npz, key):
    """Load an object array from npz and return as list of 2xN arrays (y; x)."""
    if npz is None or key not in npz:
        return None
    arr = npz[key]
    # some NumPy versions wrap as object, some as 1D object array
    if not isinstance(arr, np.ndarray):
        return None
    return [arr[i] for i in range(arr.shape[0])]

# -------------------- MAIN --------------------
if __name__ == '__main__':
    print('qc check')
    
    
    
# %%
_ensure_dir(OUTDIR)

# ---- Load required artifacts ----
ops = _load_npy(os.path.join(OUTPUT_PATH, "ops.npy"), allow_pickle=True)
if isinstance(ops, np.ndarray) and ops.dtype == object:
    ops = ops.item()

mean_img = None
if isinstance(ops, dict):
    # prefer Suite2p enhanced mean if available
    mean_img = None
    if ops.get("meanImgE") is not None:
        mean_img = ops["meanImgE"]
    elif ops.get("meanImg") is not None:
        mean_img = ops["meanImg"]


labels = _load_npy(os.path.join(QC_RESULTS, "masks.npy"))
stat   = _load_npy(os.path.join(QC_RESULTS, "stat.npy"), allow_pickle=True)
flags  = _load_npy(os.path.join(QC_RESULTS, "qc_flags.npy"), allow_pickle=True)
tags   = _load_npy(os.path.join(QC_RESULTS, "qc_tags.npy"), allow_pickle=True)
metrics= _load_npy(os.path.join(QC_RESULTS, "qc_metrics.npy"), allow_pickle=True)
kept_orig = _load_npy(os.path.join(QC_RESULTS, "kept_idx_orig.npy"))
kept_seq  = _load_npy(os.path.join(QC_RESULTS, "kept_idx_seq.npy"))

# Optional geometry + rings
geom_npz_path  = os.path.join(QC_RESULTS, "roi_geometry.npz")
rings_npz_path = os.path.join(QC_RESULTS, "neuropil_rings.npz")
geom_npz  = np.load(geom_npz_path, allow_pickle=True) if os.path.exists(geom_npz_path) else None
rings_npz = np.load(rings_npz_path, allow_pickle=True) if os.path.exists(rings_npz_path) else None

roi_pixels = _safe_object_array(geom_npz, "roi_pixels")
roi_core   = _safe_object_array(geom_npz, "roi_core_pixels")
roi_edge   = _safe_object_array(geom_npz, "roi_edge_pixels")
rings_list = _safe_object_array(rings_npz, "neuropil_pixels")

# ---- Basic shape checks ----
if labels is None:
    print("ERROR: masks.npy not found.", file=sys.stderr); sys.exit(1)
Ly, Lx = labels.shape
n_kept = len(np.unique(labels)) - 1  # background is 0

print(f"FOV: {Ly} x {Lx}")
print(f"Kept ROIs (nonzero labels): {n_kept}")

if stat is not None:
    print(f"stat.npy (kept) length: {len(stat)}")
    if len(stat) != n_kept:
        print("WARNING: stat.npy length != number of nonzero labels")

# ---- 1) Overlay of all ROI outlines ----
overlay_path = os.path.join(OUTDIR, "roi_overlay.png")
_plot_overlay(mean_img, labels, overlay_path, title="QC Kept ROIs")
print(f"Saved: {overlay_path}")

# ---- 2) Sample gallery with core/edge/ring overlays ----
roi_ids = np.unique(labels); roi_ids = roi_ids[roi_ids > 0]
if roi_ids.size > 0:
    grid_n = min(24, roi_ids.size)  # 6x4 grid
    stride = max(1, roi_ids.size // grid_n)
    sample_ids = roi_ids[::stride][:grid_n]
    ncols = 6
    nrows = math.ceil(sample_ids.size / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6*ncols, 2.6*nrows))
    axes = np.atleast_2d(axes).ravel()

    for ax, rid in zip(axes, sample_ids):
        core_xy = None; edge_xy = None; ring_xy = None
        if roi_core is not None:
            core_xy = (roi_core[rid-1][0], roi_core[rid-1][1])  # stored in kept order; label ids start at 1
        if roi_edge is not None:
            edge_xy = (roi_edge[rid-1][0], roi_edge[rid-1][1])
        if rings_list is not None:
            ring_xy = (rings_list[rid-1][0], rings_list[rid-1][1])
        _draw_roi_panel(ax, rid, labels, core_xy, edge_xy, ring_xy, mean_img)
    # turn off any leftover axes
    for k in range(sample_ids.size, axes.size):
        axes[k].axis("off")
    fig.suptitle("Sample ROI gallery (outline + core/edge/ring)", y=0.98)
    fig.tight_layout()
    gallery_path = os.path.join(OUTDIR, "roi_gallery.png")
    fig.savefig(gallery_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {gallery_path}")

# ---- 3) Flag percentages bar chart ----
if isinstance(flags, dict) or (isinstance(flags, np.ndarray) and flags.dtype == object):
    # flags saved via np.save(dict) loads as object array -> .item()
    if not isinstance(flags, dict):
        try:
            flags = flags.item()
        except Exception:
            flags = {}
if isinstance(flags, dict) and len(flags):
    names, pcts = [], []
    total = None
    for k, v in flags.items():
        v = np.asarray(v).astype(bool)
        if total is None: total = v.size
        names.append(k)
        pcts.append(100.0 * v.sum() / max(1, v.size))
    plt.figure(figsize=(6, 3.2))
    plt.bar(range(len(names)), pcts)
    plt.xticks(range(len(names)), names, rotation=30, ha='right')
    plt.ylabel('% of ROIs flagged')
    plt.title('QC Flags')
    plt.tight_layout()
    flag_path = os.path.join(OUTDIR, "flag_bars.png")
    plt.savefig(flag_path, dpi=200)
    plt.close()
    print(f"Saved: {flag_path}")

# ---- 4) Geometry histograms (if metrics present) ----
if isinstance(metrics, dict) or (isinstance(metrics, np.ndarray) and metrics.dtype == object):
    if not isinstance(metrics, dict):
        try:
            metrics = metrics.item()
        except Exception:
            metrics = {}
if isinstance(metrics, dict) and "area_px" in metrics and "n_components" in metrics:
    area = np.asarray(metrics["area_px"])
    comps = np.asarray(metrics["n_components"])
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(area[~np.isnan(area)], bins=40)
    plt.xlabel("area_px"); plt.ylabel("count"); plt.title("Area (kept)")
    plt.subplot(1, 2, 2)
    plt.hist(comps[~np.isnan(comps)], bins=np.arange(1, comps.max()+2)-0.5)
    plt.xlabel("n_components"); plt.title("Connected components (kept)")
    plt.tight_layout()
    hist_path = os.path.join(OUTDIR, "geometry_hist.png")
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"Saved: {hist_path}")

print("QC quickcheck complete.")


