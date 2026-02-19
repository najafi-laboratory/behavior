import os
import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation, binary_erosion, sobel, gaussian_filter1d
import matplotlib.pyplot as plt
import yaml
from math import pi
from skimage.morphology import skeletonize
from collections import Counter
import xml.etree.ElementTree as ET
from suite2p.extraction.dcnv import oasis, preprocess
import pickle
import pandas as pd
from scipy.ndimage import percentile_filter
from scipy.signal import savgol_filter

# ==================== CONFIG & VALIDATION ====================
def load_cfg_yaml(path: str) -> Dict[str, Any]:
    print(f"Loading config from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    print(f"Config loaded successfully with {len(cfg)} sections")
    return cfg

def validate_cfg(cfg: Dict[str, Any]) -> None:
    print("Validating configuration...")
    required_sections = {
        'paths': ['plane_dir'],
        'io': ['memmap', 'copy_on_load'],
        'selection': ['only_iscell', 'iscell_prob_min'],
        'acq': ['fs'],
        'neuropil': ['enable', 'bounds', 'lam', 'fallback_alpha'],
        'baseline': ['win_s', 'percentile', 'smooth_sigma_s', 'smooth_sigma_divisor', 'f0_epsilon'],
        'outliers': ['enable', 'z_thr'],
        'anatomy': ['prefer_channel', 'use_max_proj_if_available'],
        'labeling': ['soma_area_min', 'soma_area_max', 'soma_elongation_max', 'soma_circularity_min', 'soma_iou_min'],
        'cellpose': ['enable', 'model', 'diam', 'gpu'],
        'overlay': ['bg_pmin', 'bg_pmax', 'alpha', 'line_width', 'color_soma', 'color_process', 'fig_size', 'dpi', 'overlay_filename', 'save_dir'],
        'review': ['max_points_full', 'detail_segments', 'fig_width', 'row_height', 'dpi', 'save_dir']
    }
    
    missing = []
    for sect, keys in required_sections.items():
        if sect not in cfg:
            missing.append(f"section '{sect}'")
            continue
        for k in keys:
            if k not in cfg[sect]:
                missing.append(f"cfg['{sect}']['{k}']")
    
    if missing:
        raise KeyError(f"Missing required config items: {missing}")
    print(f"Configuration validation passed ({len(required_sections)} sections checked)")

# ==================== LOADING FUNCTIONS ====================
def _load_npy_safe(path: str, memmap: bool) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        print(f"  File not found: {os.path.basename(path)}")
        return None
    
    print(f"  Loading {os.path.basename(path)} (memmap={memmap})...")
    with open(path, "rb") as f:
        arr = np.load(f, allow_pickle=True, mmap_mode=('r' if memmap else None))
        if memmap:
            result = arr
            print(f"    memmap view: {arr.shape} {arr.dtype}")
        else:
            result = np.array(arr)
            print(f"    in-memory copy: {result.shape} {result.dtype}")
        return result


# ==================== XML METADATA LOADING ====================
# Replace the PVStateShard search section in load_imaging_metadata function:

# Replace the load_imaging_metadata function with this cleaner version:

# Replace the load_imaging_metadata function with this corrected version:

def load_imaging_metadata(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load imaging metadata from ScanImage XML file with exact PVStateShard extraction"""
    print("\n=== LOADING IMAGING METADATA ===")
    
    plane_dir = cfg["paths"]["plane_dir"]
    
    # Find XML file - look for .xml files in parent directories
    search_dirs = [
        plane_dir,  # Start in suite2p plane directory
        os.path.dirname(plane_dir),  # Parent (suite2p)
        os.path.dirname(os.path.dirname(plane_dir)),  # Grandparent (session folder)
    ]
    
    xml_file = None
    for search_dir in search_dirs:
        print(f"  Searching for XML files in: {search_dir}")
        if os.path.exists(search_dir):
            xml_files = [f for f in os.listdir(search_dir) if f.endswith('.xml')]
            if xml_files:
                # Prefer files with session name pattern
                session_xml = [f for f in xml_files if len(f) > 20]  # Likely session files
                if session_xml:
                    xml_file = os.path.join(search_dir, session_xml[0])
                    print(f"    Found session XML: {session_xml[0]}")
                    break
                else:
                    xml_file = os.path.join(search_dir, xml_files[0])
                    print(f"    Found XML file: {xml_files[0]}")
                    break
    
    if xml_file is None:
        # Fallback: construct expected path from plane_dir
        session_folder = os.path.dirname(os.path.dirname(plane_dir))
        session_name = os.path.basename(session_folder)
        expected_xml = os.path.join(session_folder, f"{session_name}.xml")
        
        print(f"  No XML found in search dirs, trying expected path: {expected_xml}")
        if os.path.exists(expected_xml):
            xml_file = expected_xml
        else:
            raise FileNotFoundError(f"Could not find imaging XML file. Searched: {search_dirs}")
    
    print(f"Loading metadata from: {xml_file}")
    
    # Parse XML
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        print(f"  XML root tag: {root.tag}")
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML file: {e}")
    
    # Initialize metadata dictionary
    metadata = {
        'xml_file': xml_file,
        'microns_per_pixel_x': None,
        'microns_per_pixel_y': None,
        'microns_per_pixel_z': None,
        'objective_mag': None,
        'objective_na': None,
        'optical_zoom': None,
        'pixel_size_um': None,  # Will be computed
        'fov_size_um': None,    # Will be computed from image dimensions
    }
    
    print("  Searching for PVStateShard structure...")
    
    # Since PVScan is the root, search for PVStateShard directly within root
    pv_state_values = []
    
    if root.tag == 'PVScan':
        print(f"    Root is PVScan, searching for PVStateShard children...")
        
        # Search for PVStateShard elements directly under root, avoiding Sequence sections
        for child in root:
            if child.tag == 'PVStateShard':
                print(f"      Found PVStateShard directly under root")
                shard_values = child.findall('./PVStateValue')
                pv_state_values.extend(shard_values)
                print(f"        Contains {len(shard_values)} PVStateValue entries")
            elif child.tag != 'Sequence':
                # Look one level deeper, but skip Sequence elements
                nested_shards = child.findall('./PVStateShard')
                if len(nested_shards) > 0:
                    print(f"      Found {len(nested_shards)} PVStateShard in {child.tag}")
                    for shard in nested_shards:
                        shard_values = shard.findall('./PVStateValue')
                        pv_state_values.extend(shard_values)
                        print(f"        Shard contains {len(shard_values)} PVStateValue entries")
    else:
        print(f"    Root is not PVScan (is {root.tag}), searching for PVScan elements...")
        # Fallback: search for PVScan elements within root
        pv_scan_elements = root.findall('.//PVScan')
        print(f"    Found {len(pv_scan_elements)} PVScan element(s)")
        
        if len(pv_scan_elements) > 0:
            # Use first PVScan element
            pv_scan = pv_scan_elements[0]
            print(f"    Using first PVScan element")
            
            # Search for PVStateShard elements, avoiding Sequence sections
            for child in pv_scan:
                if child.tag == 'PVStateShard':
                    print(f"      Found PVStateShard in PVScan")
                    shard_values = child.findall('./PVStateValue')
                    pv_state_values.extend(shard_values)
                    print(f"        Contains {len(shard_values)} PVStateValue entries")
                elif child.tag != 'Sequence':
                    # Look one level deeper, but skip Sequence elements
                    nested_shards = child.findall('./PVStateShard')
                    if len(nested_shards) > 0:
                        print(f"      Found {len(nested_shards)} PVStateShard in {child.tag}")
                        for shard in nested_shards:
                            shard_values = shard.findall('./PVStateValue')
                            pv_state_values.extend(shard_values)
                            print(f"        Shard contains {len(shard_values)} PVStateValue entries")
    
    if len(pv_state_values) == 0:
        print("    No PVStateShard found, searching all PVStateValue elements...")
        # Fallback: find all PVStateValue elements, build parent map to avoid Sequence
        parent_map = {}
        for parent in root.iter():
            for child in parent:
                parent_map[child] = parent
        
        all_pv_values = root.findall('.//PVStateValue')
        for pv in all_pv_values:
            parent = parent_map.get(pv)
            if parent is not None and parent.tag != 'Sequence':
                pv_state_values.append(pv)
        
        print(f"    Found {len(pv_state_values)} PVStateValue entries (excluding Sequence)")
    
    print(f"    Total PVStateValue entries to process: {len(pv_state_values)}")
    
    if len(pv_state_values) == 0:
        print("    WARNING: No PVStateValue entries found")
        print("    XML structure debug info:")
        for i, child in enumerate(root):
            print(f"      Root child {i}: {child.tag}")
            if child.tag == 'PVStateShard':
                pv_vals = child.findall('./PVStateValue')
                print(f"        PVStateShard has {len(pv_vals)} PVStateValue entries")
                if len(pv_vals) > 0:
                    for k, pv in enumerate(pv_vals[:3]):  # Show first 3
                        key_elem = pv.find('key')
                        key_text = key_elem.text if key_elem is not None else 'NO_KEY'
                        print(f"          PV {k}: key='{key_text}'")
            elif child.tag != 'Sequence':
                print(f"        Non-Sequence child, checking for nested PVStateShard...")
                nested_shards = child.findall('./PVStateShard')
                print(f"          Found {len(nested_shards)} nested PVStateShard")
    
    # Process each PVStateValue
    # Process each PVStateValue
    for i, pv in enumerate(pv_state_values):
        print(f"      Processing PVStateValue {i}:")
        print(f"        Tag: {pv.tag}")
        print(f"        Attributes: {pv.attrib}")
        print(f"        Children: {[child.tag for child in pv]}")
        
        # Get key from attributes
        key = pv.attrib.get('key')
        if key is None:
            print(f"        No key found, skipping")
            continue
        
        print(f"        Key: '{key}'")
        
        # Check if value is in attributes (simple case)
        if 'value' in pv.attrib:
            value_text = pv.attrib['value']
            print(f"        Value from attrib: '{value_text}'")
            
            # Parse specific parameters
            if key == 'objectiveLens':
                try:
                    # Handle formats like "16x" or "16"
                    mag_str = value_text.replace('x', '').strip()
                    metadata['objective_mag'] = float(mag_str)
                    print(f"        -> Set objective_mag: {metadata['objective_mag']}x")
                except (ValueError, TypeError) as e:
                    print(f"        -> Error parsing objectiveLens: {e}")
            
            elif key == 'objectiveLensMag':
                try:
                    metadata['objective_mag'] = float(value_text)
                    print(f"        -> Set objective_mag: {metadata['objective_mag']}x")
                except (ValueError, TypeError) as e:
                    print(f"        -> Error parsing objectiveLensMag: {e}")
            
            elif key == 'objectiveLensNA':
                try:
                    metadata['objective_na'] = float(value_text)
                    print(f"        -> Set objective_na: {metadata['objective_na']}")
                except (ValueError, TypeError) as e:
                    print(f"        -> Error parsing objectiveLensNA: {e}")
            
            elif key == 'opticalZoom':
                try:
                    metadata['optical_zoom'] = float(value_text)
                    print(f"        -> Set optical_zoom: {metadata['optical_zoom']}x")
                except (ValueError, TypeError) as e:
                    print(f"        -> Error parsing opticalZoom: {e}")
            
            else:
                print(f"        -> Unrecognized key, skipping")
        
        # Check if value is in children (complex case like micronsPerPixel)
        elif key == 'micronsPerPixel':
            print(f"        Found micronsPerPixel - processing IndexedValue children...")
            # Look for IndexedValue children
            indexed_values = [child for child in pv if child.tag == 'IndexedValue']
            print(f"          Found {len(indexed_values)} IndexedValue children")
            
            for iv in indexed_values:
                print(f"            IndexedValue attrib: {iv.attrib}")
                print(f"            IndexedValue children: {[child.tag for child in iv]}")
                
                # Handle string-based axis indices - NO INTEGER CONVERSION
                idx_str = iv.attrib.get('index')
                val_str = iv.attrib.get('value')
                
                if idx_str is not None and val_str is not None:
                    try:
                        val = float(val_str)
                        
                        # Map string indices directly - NO int() conversion
                        if idx_str == 'XAxis':
                            metadata['microns_per_pixel_x'] = val
                            print(f"            X-axis ({idx_str}): {val} um/pixel")
                        elif idx_str == 'YAxis':
                            metadata['microns_per_pixel_y'] = val
                            print(f"            Y-axis ({idx_str}): {val} um/pixel")
                        elif idx_str == 'ZAxis':
                            metadata['microns_per_pixel_z'] = val
                            print(f"            Z-axis ({idx_str}): {val} um/pixel")
                        else:
                            print(f"            Unknown axis index {idx_str}: {val}")
                    except (ValueError, TypeError) as e:
                        print(f"            Error parsing IndexedValue value: {e}")
                else:
                    print(f"            Could not find index/value attributes")
    # Calculate and verify pixel size
    # Calculate and verify pixel size
    print("\n  PIXEL SIZE CALCULATION AND VERIFICATION:")
    
    # First, use the direct microns_per_pixel values
    if metadata['microns_per_pixel_x'] is not None:
        metadata['pixel_size_um'] = metadata['microns_per_pixel_x']
        print(f"    Direct from XML - pixel size: {metadata['pixel_size_um']:.6f} um/pixel")
        
        # Verify Y-axis matches X-axis
        if metadata['microns_per_pixel_y'] is not None:
            if abs(metadata['microns_per_pixel_x'] - metadata['microns_per_pixel_y']) > 1e-6:
                print(f"    WARNING: X and Y pixel sizes differ!")
                print(f"        X: {metadata['microns_per_pixel_x']:.6f} um")
                print(f"        Y: {metadata['microns_per_pixel_y']:.6f} um")
            else:
                print(f"    X and Y pixel sizes match: {metadata['microns_per_pixel_x']:.6f} um")
    
    # Calculate theoretical pixel size from zoom and objective
    if metadata['optical_zoom'] is not None and metadata['objective_mag'] is not None:
        print(f"\n    Theoretical calculation using:")
        print(f"      Optical zoom: {metadata['optical_zoom']}x")
        print(f"      Objective mag: {metadata['objective_mag']}x")
        print(f"      Total magnification: {metadata['optical_zoom'] * metadata['objective_mag']}x")
        
        # The formula typically used is: pixel_size = FOV_size / (pixels * total_magnification)
        # But we need to determine the correct baseline or field of view
        
        # Common baselines for different systems
        baseline_candidates = [
            # Common values for different microscope setups
            35.7,   # 35.7 um at 1x zoom, 1x objective (common for many 2P systems)
            30.0,   # 30 um baseline
            40.0,   # 40 um baseline
            50.0,   # 50 um baseline
            24.0,   # 24 um baseline
            20.0,   # 20 um baseline
        ]
        
        print(f"\n    Testing baseline pixel sizes (assuming 512x512 pixels):")
        print(f"    Formula: pixel_size = baseline_um / total_magnification")
        
        best_match = None
        best_diff = float('inf')
        
        for baseline in baseline_candidates:
            calc_pixel_size = baseline / (metadata['optical_zoom'] * metadata['objective_mag'])
            print(f"      Baseline {baseline:.1f} um -> {calc_pixel_size:.6f} um/pixel")
            
            # Compare with direct measurement if available
            if metadata['pixel_size_um'] is not None:
                diff = abs(calc_pixel_size - metadata['pixel_size_um'])
                diff_pct = 100 * diff / metadata['pixel_size_um']
                print(f"        Difference from XML: {diff:.6f} um ({diff_pct:.1f}%)")
                
                if diff < best_diff:
                    best_diff = diff
                    best_match = (baseline, calc_pixel_size)
        
        if best_match is not None and metadata['pixel_size_um'] is not None:
            baseline, calc_size = best_match
            print(f"\n    Best match: baseline = {baseline:.1f} um")
            print(f"      Calculated: {calc_size:.6f} um/pixel")
            print(f"      XML direct: {metadata['pixel_size_um']:.6f} um/pixel")
            diff_pct = 100 * best_diff / metadata['pixel_size_um']
            print(f"      Difference: {best_diff:.6f} um ({diff_pct:.1f}%)")
            
            # Store the baseline for future reference
            metadata['baseline_pixel_size_um'] = baseline
            metadata['calculated_pixel_size_um'] = calc_size
            
            # Warning if difference is large
            if diff_pct > 10:
                print(f"    *** WARNING: Large discrepancy ({diff_pct:.1f}%) between direct and calculated values ***")
                print(f"        This suggests the baseline assumption may be incorrect")
                print(f"        Using direct XML value: {metadata['pixel_size_um']:.6f} um/pixel")
            else:
                print(f"    Good agreement between direct and calculated values")
        
        # Try reverse calculation to find what baseline would match
        if metadata['pixel_size_um'] is not None:
            implied_baseline = metadata['pixel_size_um'] * (metadata['optical_zoom'] * metadata['objective_mag'])
            print(f"\n    Reverse calculation:")
            print(f"      To get {metadata['pixel_size_um']:.6f} um/pixel with {metadata['optical_zoom']}x zoom and {metadata['objective_mag']}x objective:")
            print(f"      Required baseline = {implied_baseline:.1f} um per unit magnification")
            
            # Calculate the FOV at 1x zoom, 1x objective (assuming 512x512 pixels)
            implied_fov = implied_baseline * 512 / metadata['objective_mag']
            print(f"      This implies FOV = {implied_fov:.1f} um at 1x zoom, 1x objective")
            print(f"      (Your actual specs: 1141 um FOV, 2.229 um/pixel at 1x zoom)")
            
            # Check agreement with your known specifications
            actual_fov_1x = 1141.0  # From your specs
            actual_pixel_1x = 2.229  # From your specs
            
            fov_diff = abs(implied_fov - actual_fov_1x)
            fov_diff_pct = 100 * fov_diff / actual_fov_1x
            
            print(f"      Agreement check:")
            print(f"        Calculated FOV at 1x: {implied_fov:.1f} um")
            print(f"        Your actual FOV at 1x: {actual_fov_1x:.1f} um") 
            print(f"        Difference: {fov_diff:.1f} um ({fov_diff_pct:.1f}%)")
            
            if fov_diff_pct < 1:
                print(f"        ✓ Excellent agreement - XML values are correct!")
            elif fov_diff_pct < 5:
                print(f"        ✓ Good agreement - XML values are reliable")
            else:
                print(f"        ⚠ Poor agreement - check scope specifications or XML parsing")
                
            # Store the verified values
            metadata['fov_1x_um'] = actual_fov_1x
            metadata['pixel_1x_um'] = actual_pixel_1x
            metadata['total_magnification'] = metadata['optical_zoom'] * metadata['objective_mag']
    
    # Summary of extracted metadata
    print(f"\n  METADATA EXTRACTION SUMMARY:")
    print(f"    XML file: {os.path.basename(xml_file)}")
    
    for key, value in metadata.items():
        if key == 'xml_file':
            continue
        
        if value is not None:
            if 'microns' in key or 'pixel_size' in key:
                print(f"    {key}: {value:.6f} um")
            elif 'na' in key:
                print(f"    {key}: {value:.2f}")
            elif 'mag' in key or 'zoom' in key:
                print(f"    {key}: {value}x")
            else:
                print(f"    {key}: {value}")
        else:
            print(f"    {key}: NOT FOUND")
    
    # Warnings for missing critical parameters
    missing_params = []
    if metadata['microns_per_pixel_x'] is None:
        missing_params.append('microns_per_pixel_x')
    if metadata['optical_zoom'] is None:
        missing_params.append('optical_zoom')
    if metadata['objective_mag'] is None:
        missing_params.append('objective_mag')
    
    if missing_params:
        print(f"\n  WARNING: Missing parameters: {missing_params}")
        print(f"      Some size-dependent calculations may be inaccurate")
        
        # Provide fallback values with warning
        if metadata['microns_per_pixel_x'] is None:
            metadata['microns_per_pixel_x'] = 0.5  # Conservative estimate
            metadata['microns_per_pixel_y'] = 0.5
            metadata['pixel_size_um'] = 0.5
            print(f"      Using fallback pixel size: 0.5 um/pixel")
    else:
        print(f"\n  All critical parameters successfully extracted")
    
    return metadata

def update_features_with_metadata(data: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """Update ROI features with physical size measurements"""
    print("\n=== UPDATING FEATURES WITH PHYSICAL SIZES ===")
    
    if 'roi_features' not in data:
        print("No ROI features found - run feature extraction first")
        return
    
    pixel_size = metadata.get('pixel_size_um', metadata.get('microns_per_pixel_x', 0.5))
    print(f"Using pixel size: {pixel_size:.3f} μm/pixel")
    
    features = data['roi_features']
    n_rois = len(features['area'])
    
    # Add physical size features
    features['area_um2'] = features['area'] * (pixel_size ** 2)
    features['major_axis_um'] = features['major_axis'] * pixel_size
    features['minor_axis_um'] = features['minor_axis'] * pixel_size
    features['thickness_um'] = features['thickness_px'] * pixel_size
    features['skeleton_length_um'] = features['skeleton_length'] * pixel_size
    
    print(f"Added physical size features:")
    valid_mask = features['area'] > 0
    if valid_mask.sum() > 0:
        print(f"  Area: [{features['area_um2'][valid_mask].min():.1f}, {features['area_um2'][valid_mask].max():.1f}] μm², median: {np.median(features['area_um2'][valid_mask]):.1f}")
        print(f"  Major axis: [{features['major_axis_um'][valid_mask].min():.1f}, {features['major_axis_um'][valid_mask].max():.1f}] μm")
        print(f"  Thickness: [{features['thickness_um'][valid_mask].min():.2f}, {features['thickness_um'][valid_mask].max():.2f}] μm")
    
    # Store metadata in data for later use
    data['imaging_metadata'] = metadata

def classify_rois_with_physical_sizes(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """Enhanced classification using physical sizes in addition to pixel-based features"""
    print("\n=== CLASSIFICATION WITH PHYSICAL SIZES ===")
    
    feats = data.get('roi_features')
    metadata = data.get('imaging_metadata')
    
    if feats is None:
        raise ValueError("Features missing - run extract_refined_roi_features first")
    if metadata is None:
        print("Warning: No imaging metadata available - using pixel-based classification only")
        return classify_rois_refined(data, cfg)
    
    pixel_size = metadata.get('pixel_size_um', 0.5)
    objective_na = metadata.get('objective_na')
    
    print("Physical size-aware classification criteria:")
    print(f"  Pixel size: {pixel_size:.3f} μm/pixel")
    if objective_na:
        print(f"  Objective NA: {objective_na:.2f}")
    
    # Physical size-based thresholds
    soma_area_um2_min = 15.0  # μm² - typical Purkinje cell soma
    soma_area_um2_max = 400.0  # μm² - upper limit
    soma_diameter_um_min = 4.0  # μm - minimum soma diameter
    soma_diameter_um_max = 25.0  # μm - maximum soma diameter
    
    dendrite_width_um_max = 3.0  # μm - dendrite thickness
    dendrite_length_um_min = 8.0  # μm - minimum dendrite length
    
    print(f"  Physical criteria:")
    print(f"    Soma area: [{soma_area_um2_min}, {soma_area_um2_max}] μm²")
    print(f"    Soma diameter: [{soma_diameter_um_min}, {soma_diameter_um_max}] μm")
    print(f"    Dendrite max width: {dendrite_width_um_max} μm")
    print(f"    Dendrite min length: {dendrite_length_um_min} μm")
    
    # Original pixel-based criteria (scaled for robustness)
    soma_aspect_max = 1.9
    soma_solidity_min = 0.86
    soma_circularity_min = 0.62
    proc_coherence_min = 0.60
    proc_circularity_max = 0.80
    
    # Extract features
    area_px = feats['area']
    area_um2 = feats['area_um2']
    aspect_ratio = feats['aspect_ratio']
    solidity = feats['solidity']
    circularity = feats['circularity']
    thickness_um = feats['thickness_um']
    major_axis_um = feats['major_axis_um']
    orientation_coh = feats['orientation_coherence']
    skeleton_len_um = feats['skeleton_length_um']
    
    n_rois = len(area_px)
    print(f"\nEvaluating {n_rois} ROIs with physical constraints...")
    
    # Physical size criteria
    soma_area_phys = (area_um2 >= soma_area_um2_min) & (area_um2 <= soma_area_um2_max)
    soma_diameter_phys = (major_axis_um >= soma_diameter_um_min) & (major_axis_um <= soma_diameter_um_max)
    
    # Shape criteria (same as before)
    soma_aspect = aspect_ratio <= soma_aspect_max
    soma_solid = solidity >= soma_solidity_min
    soma_circ = circularity >= soma_circularity_min
    
    # Combined soma criteria (physical AND shape)
    soma_all = soma_area_phys & soma_diameter_phys & soma_aspect & soma_solid & soma_circ
    
    # Process criteria with physical constraints
    proc_thin = thickness_um <= dendrite_width_um_max
    proc_long = skeleton_len_um >= dendrite_length_um_min
    proc_elongated = aspect_ratio >= 2.1
    proc_coherent = orientation_coh >= proc_coherence_min
    proc_not_lumpy = circularity <= proc_circularity_max
    
    # Process: meet physical size constraints AND (elongated OR coherent OR long skeleton)
    proc_shape_any = proc_elongated | proc_coherent | proc_long
    proc_all = proc_thin & proc_shape_any & proc_not_lumpy
    
    print(f"  Physical criteria breakdown:")
    print(f"    Soma area (physical): {soma_area_phys.sum()}/{n_rois} ({100*soma_area_phys.sum()/n_rois:.1f}%)")
    print(f"    Soma diameter (physical): {soma_diameter_phys.sum()}/{n_rois} ({100*soma_diameter_phys.sum()/n_rois:.1f}%)")
    print(f"    Process thin enough: {proc_thin.sum()}/{n_rois} ({100*proc_thin.sum()/n_rois:.1f}%)")
    print(f"    Process long enough: {proc_long.sum()}/{n_rois} ({100*proc_long.sum()/n_rois:.1f}%)")
    
    print(f"  Shape criteria breakdown:")
    print(f"    Soma aspect ratio: {soma_aspect.sum()}/{n_rois} ({100*soma_aspect.sum()/n_rois:.1f}%)")
    print(f"    Soma solidity: {soma_solid.sum()}/{n_rois} ({100*soma_solid.sum()/n_rois:.1f}%)")
    print(f"    Soma circularity: {soma_circ.sum()}/{n_rois} ({100*soma_circ.sum()/n_rois:.1f}%)")
    
    print(f"  Combined criteria:")
    print(f"    ALL soma criteria: {soma_all.sum()}/{n_rois} ({100*soma_all.sum()/n_rois:.1f}%)")
    print(f"    ALL process criteria: {proc_all.sum()}/{n_rois} ({100*proc_all.sum()/n_rois:.1f}%)")
    
    # Handle Cellpose IoU if available
    iou_soma_force = np.zeros(n_rois, dtype=bool)
    iou_process_allow = np.ones(n_rois, dtype=bool)
    
    if 'cellpose_iou' in data:
        print("  Applying Cellpose IoU gates...")
        iou = data['cellpose_iou']
        iou_soma_force = iou >= 0.35
        iou_process_allow = iou <= 0.15
        print(f"    IoU >= 0.35 (force soma): {iou_soma_force.sum()}/{n_rois} ({100*iou_soma_force.sum()/n_rois:.1f}%)")
        print(f"    IoU <= 0.15 (allow process): {iou_process_allow.sum()}/{n_rois} ({100*iou_process_allow.sum()/n_rois:.1f}%)")
    
    # Final classification
    print("  Applying final classification logic...")
    labels = []
    soma_count = 0
    process_count = 0
    uncertain_count = 0
    iou_forced_soma_count = 0
    
    for i in range(n_rois):
        if iou_soma_force[i]:
            labels.append('soma')
            soma_count += 1
            iou_forced_soma_count += 1
        elif soma_all[i]:
            labels.append('soma')
            soma_count += 1
        elif proc_all[i] and iou_process_allow[i]:
            labels.append('process')
            process_count += 1
        else:
            labels.append('uncertain')
            uncertain_count += 1
    
    data['roi_labels'] = labels
    
    # Final summary
    print(f"\nPhysical size-aware classification results:")
    total = len(labels)
    print(f"  Soma: {soma_count}/{total} ({100*soma_count/total:.1f}%)")
    if iou_forced_soma_count > 0:
        print(f"    (including {iou_forced_soma_count} IoU-forced)")
    print(f"  Process: {process_count}/{total} ({100*process_count/total:.1f}%)")
    print(f"  Uncertain: {uncertain_count}/{total} ({100*uncertain_count/total:.1f}%)")
    
    # Physical size distribution summary
    print(f"\nPhysical size distributions:")
    soma_mask = np.array(labels) == 'soma'
    proc_mask = np.array(labels) == 'process'
    
    if soma_mask.sum() > 0:
        soma_areas = area_um2[soma_mask]
        soma_diams = major_axis_um[soma_mask]
        print(f"  Soma areas: {soma_areas.mean():.1f} ± {soma_areas.std():.1f} μm² (range: {soma_areas.min():.1f}-{soma_areas.max():.1f})")
        print(f"  Soma diameters: {soma_diams.mean():.1f} ± {soma_diams.std():.1f} μm (range: {soma_diams.min():.1f}-{soma_diams.max():.1f})")
    
    if proc_mask.sum() > 0:
        proc_thickness = thickness_um[proc_mask]
        proc_lengths = skeleton_len_um[proc_mask]
        print(f"  Process thickness: {proc_thickness.mean():.2f} ± {proc_thickness.std():.2f} μm (range: {proc_thickness.min():.2f}-{proc_thickness.max():.2f})")
        proc_lengths_valid = proc_lengths[proc_lengths > 0]
        if len(proc_lengths_valid) > 0:
            print(f"  Process lengths: {proc_lengths_valid.mean():.1f} ± {proc_lengths_valid.std():.1f} μm (range: {proc_lengths_valid.min():.1f}-{proc_lengths_valid.max():.1f})")




def load_suite2p(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """STEP 1: Load raw Suite2p outputs with proper data structure handling"""
    print("\n=== LOADING SUITE2P DATA ===")
    folder = cfg["paths"]["plane_dir"]
    memmap = cfg["io"]["memmap"]
    copy_on_load = cfg["io"]["copy_on_load"]
    
    print(f"Source folder: {folder}")
    print(f"Memmap mode: {memmap}, Copy on load: {copy_on_load}")
    
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Suite2p folder not found: {folder}")
    
    data: Dict[str, Any] = {}
    
    # Load core arrays (these are 2D arrays)
    data["F"]      = _load_npy_safe(os.path.join(folder, "F.npy"), memmap)
    data["Fneu"]   = _load_npy_safe(os.path.join(folder, "Fneu.npy"), memmap)
    data["spks"]   = _load_npy_safe(os.path.join(folder, "spks.npy"), memmap)
    data["iscell"] = _load_npy_safe(os.path.join(folder, "iscell.npy"), memmap=False)
    
    # Load ops (dictionary) - FIXED: handle properly
    ops_path = os.path.join(folder, "ops.npy")
    if os.path.exists(ops_path):
        print(f"  Loading ops.npy...")
        ops_raw = np.load(ops_path, allow_pickle=True)
        
        # ops is usually saved as a numpy array containing a dict
        if isinstance(ops_raw, np.ndarray):
            if ops_raw.ndim == 0:  # 0-dimensional array containing a dict
                data["ops"] = ops_raw.item()
                print(f"    Extracted ops dict from 0-d array: {type(data['ops'])}")
            elif len(ops_raw) == 1:  # 1-element array containing a dict
                data["ops"] = ops_raw[0]
                print(f"    Extracted ops dict from 1-element array: {type(data['ops'])}")
            else:
                print(f"    WARNING: Unexpected ops array shape: {ops_raw.shape}")
                data["ops"] = ops_raw
        else:
            data["ops"] = ops_raw
            print(f"    Loaded ops directly: {type(data['ops'])}")
        
        # Verify it's a dictionary
        if isinstance(data["ops"], dict):
            print(f"    ops dict contains {len(data['ops'])} keys")
        else:
            print(f"    WARNING: ops is not a dict, it's {type(data['ops'])}")
    else:
        print(f"  ops.npy not found")
        data["ops"] = None
    
    # Load stat (list of dictionaries) - FIXED: handle properly  
    stat_path = os.path.join(folder, "stat.npy")
    if os.path.exists(stat_path):
        print(f"  Loading stat.npy...")
        stat_raw = np.load(stat_path, allow_pickle=True)
        
        # stat is usually saved as a numpy object array containing dicts
        if isinstance(stat_raw, np.ndarray):
            if stat_raw.dtype == object:
                data["stat"] = list(stat_raw)  # Convert object array to list
                print(f"    Converted stat object array to list: {len(data['stat'])} entries")
            else:
                print(f"    WARNING: stat array has unexpected dtype: {stat_raw.dtype}")
                data["stat"] = stat_raw
        else:
            data["stat"] = stat_raw
            print(f"    Loaded stat directly: {type(data['stat'])}")
        
        # Verify it's a list of dicts
        if isinstance(data["stat"], list) and len(data["stat"]) > 0:
            if isinstance(data["stat"][0], dict):
                print(f"    stat list contains {len(data['stat'])} ROI dicts")
                print(f"    First ROI keys: {list(data['stat'][0].keys())[:5]}...")  # Show first few keys
            else:
                print(f"    WARNING: stat[0] is not a dict, it's {type(data['stat'][0])}")
        else:
            print(f"    WARNING: stat is not a list or is empty")
    else:
        print(f"  stat.npy not found")
        data["stat"] = None
    
    # Validation
    if data["F"] is None or data["Fneu"] is None:
        raise RuntimeError("Missing required F or Fneu arrays.")
    
    print(f"Initial shapes: F{data['F'].shape}, Fneu{data['Fneu'].shape}")
    
    # Shape matching
    if data["F"].shape != data["Fneu"].shape:
        print("  Shape mismatch detected, aligning arrays...")
        n = min(data["F"].shape[0], data["Fneu"].shape[0])
        t = min(data["F"].shape[1], data["Fneu"].shape[1])
        data["F"] = data["F"][:n, :t]
        data["Fneu"] = data["Fneu"][:n, :t]
        print(f"  Aligned to: F{data['F'].shape}, Fneu{data['Fneu'].shape}")
    
    # Optional copy from memmap
    if memmap and copy_on_load:
        print("  Converting memmap to in-memory arrays...")
        for k in ("F", "Fneu", "spks"):
            if data.get(k) is not None:
                old_type = type(data[k])
                data[k] = np.array(data[k])
                print(f"    {k}: {old_type} to {type(data[k])}")
    
    # Summary
    n_rois, n_timepoints = data["F"].shape
    duration_s = n_timepoints / cfg["acq"]["fs"]
    print(f"Loaded {n_rois} ROIs x {n_timepoints} timepoints ({duration_s:.1f}s)")
    
    # Verify consistency between arrays and metadata
    if data["stat"] is not None:
        if len(data["stat"]) != n_rois:
            print(f"  WARNING: stat length ({len(data['stat'])}) != F ROIs ({n_rois})")
    
    if data["iscell"] is not None:
        if len(data["iscell"]) != n_rois:
            print(f"  WARNING: iscell length ({len(data['iscell'])}) != F ROIs ({n_rois})")
    
    return data






def subset_cells(data: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """STEP 2: Filter to cells only (optional)"""
    print("\n=== CELL FILTERING ===")
    
    if not cfg["selection"]["only_iscell"]:
        print("Cell filtering disabled in config")
        return data
    
    if data.get("iscell") is None:
        print("No iscell array found, skipping filtering")
        return data
    
    iscell = np.array(data["iscell"])
    prob_min = cfg["selection"]["iscell_prob_min"]
    
    print(f"iscell array shape: {iscell.shape}")
    print(f"Probability threshold: {prob_min}")
    
    # Generate mask
    if iscell.ndim != 2 or iscell.shape[1] < 2:
        print("  Using binary iscell column only")
        mask = iscell[:, 0] > 0
    else:
        print("  Using both binary and probability columns")
        mask = (iscell[:, 0] > 0) & (iscell[:, 1] >= prob_min)
    
    n_total = len(mask)
    n_kept = mask.sum()
    
    if n_kept == 0:
        raise RuntimeError("No ROIs passed iscell filter.")
    
    print(f"Filter results: {n_kept}/{n_total} ROIs kept ({100*n_kept/n_total:.1f}%)")
    
    # Apply mask to arrays
    arrays_filtered = []
    for k in ("F", "Fneu", "spks"):
        if data.get(k) is not None:
            old_shape = data[k].shape
            data[k] = data[k][mask]
            arrays_filtered.append(f"{k}: {old_shape} to {data[k].shape}")
    
    # Subset stat list
    if data.get('stat') is not None:
        old_len = len(data['stat'])
        stat_arr = np.array(data['stat'], dtype=object)
        data['stat'] = list(stat_arr[mask])
        arrays_filtered.append(f"stat: {old_len} to {len(data['stat'])}")
    
    for af in arrays_filtered:
        print(f"  {af}")
    
    data["roi_mask"] = mask
    print(f"Cell filtering complete")
    return data

# ==================== ANATOMY & FEATURES ====================
def select_anatomical_image(data: Dict[str, Any], cfg: Dict[str, Any]) -> np.ndarray:
    """STEP 3A: Choose anatomical reference image"""
    print("\n=== SELECTING ANATOMICAL IMAGE ===")
    
    ops = data.get("ops")
    if ops is None:
        raise ValueError("ops missing from data")
    
    if isinstance(ops, np.ndarray):
        ops = ops.item()
        print("Extracted ops dict from numpy array")
    
    if 'anatomy' not in cfg:
        raise KeyError("Config missing 'anatomy' section")
    
    a_cfg = cfg['anatomy']
    prefer_ch = a_cfg['prefer_channel']
    use_max_proj = a_cfg['use_max_proj_if_available']
    
    print(f"Preference: channel {prefer_ch}, max_proj fallback: {use_max_proj}")
    print(f"Available ops keys: {list(ops.keys())}")
    
    A = None
    source = "unknown"
    
    # Channel 2 preference
    if prefer_ch == 2:
        print("  Checking for channel 2 images...")
        if 'meanImg_chan2' in ops:
            A = ops['meanImg_chan2']
            source = "meanImg_chan2"
        elif 'meanImg_chan2_corrected' in ops:
            A = ops['meanImg_chan2_corrected']
            source = "meanImg_chan2_corrected"
        
        if A is not None:
            print(f"  Found {source}")
    
    # Max proj fallback
    if A is None and use_max_proj:
        print("  Checking for max projection images...")
        if 'max_proj_chan2' in ops:
            A = ops['max_proj_chan2']
            source = "max_proj_chan2"
        elif 'max_proj' in ops:
            A = ops['max_proj']
            source = "max_proj"
        
        if A is not None:
            print(f"  Found {source}")
    
    # Final fallback
    if A is None:
        print("  Using fallback meanImg...")
        if 'meanImg' in ops:
            A = ops['meanImg']
            source = "meanImg"
        
        if A is not None:
            print(f"  Found {source}")
    
    if A is None:
        raise ValueError(f"No anatomical image found. Available: {list(ops.keys())}")
    
    A = np.asarray(A, dtype=np.float32)
    data['anatomy_image'] = A
    
    print(f"Selected {source}: {A.shape} {A.dtype}")
    print(f"  Intensity range: [{A.min():.1f}, {A.max():.1f}], mean: {A.mean():.1f}")
    
    return A

def _roi_pixels_from_stat(stat_entry: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ypix = np.array(stat_entry['ypix'], dtype=np.int32)
    xpix = np.array(stat_entry['xpix'], dtype=np.int32)
    lam = np.array(stat_entry.get('lam', np.ones_like(ypix)), dtype=np.float32)
    if lam.shape[0] != ypix.shape[0]:
        lam = np.ones_like(ypix, dtype=np.float32)
    return ypix, xpix, lam

def _clip_roi_pixels_to_image(ypix: np.ndarray, xpix: np.ndarray, lam: np.ndarray, H: int, W: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = (xpix >= 0) & (xpix < W) & (ypix >= 0) & (ypix < H)
    if not m.all():
        ypix, xpix, lam = ypix[m], xpix[m], lam[m]
    return ypix, xpix, lam

def extract_all_roi_features(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """STEP 3B: Extract shape/intensity features for each ROI"""
    print("\n=== EXTRACTING ROI FEATURES ===")
    
    stat_list = data.get('stat')
    if stat_list is None:
        raise ValueError("stat list missing from data")
    
    n_stat = len(stat_list)
    n_rois = data['F'].shape[0]
    
    print(f"stat entries: {n_stat}, F ROIs: {n_rois}")
    
    if n_stat != n_rois:
        raise ValueError(f"stat length ({n_stat}) != F ROIs ({n_rois})")
    
    # Get anatomy image
    A = data.get('anatomy_image')
    if A is None:
        print("  Anatomy image not cached, selecting...")
        A = select_anatomical_image(data, cfg)
    else:
        print(f"  Using cached anatomy image: {A.shape}")
    
    # Global stats
    g_mean = float(np.nanmean(A))
    g_std = float(np.nanstd(A) + 1e-9)
    print(f"Global anatomy stats: mean={g_mean:.1f}, std={g_std:.1f}")
    
    # Initialize feature arrays
    features = {
        'area': np.zeros(n_rois, dtype=np.int32),
        'centroid_x': np.zeros(n_rois, dtype=np.float32),
        'centroid_y': np.zeros(n_rois, dtype=np.float32),
        'major_axis': np.zeros(n_rois, dtype=np.float32),
        'minor_axis': np.zeros(n_rois, dtype=np.float32),
        'elongation': np.zeros(n_rois, dtype=np.float32),
        'circ_ratio': np.zeros(n_rois, dtype=np.float32),
        'inten_mean': np.zeros(n_rois, dtype=np.float32),
        'inten_median': np.zeros(n_rois, dtype=np.float32),
        'inten_std': np.zeros(n_rois, dtype=np.float32),
        'inten_contrast_z': np.zeros(n_rois, dtype=np.float32)
    }
    
    oob_count = 0
    zero_area_count = 0
    small_roi_count = 0
    
    print("  Processing ROIs...")
    for i, st in enumerate(stat_list):
        if i % 100 == 0 and i > 0:
            print(f"    Processed {i}/{n_rois} ROIs...")
        
        ypix, xpix, lam = _roi_pixels_from_stat(st)
        original_size = ypix.size
        
        # Bounds clipping
        ypix, xpix, lam = _clip_roi_pixels_to_image(ypix, xpix, lam, A.shape[0], A.shape[1])
        clipped_size = ypix.size
        
        if clipped_size < original_size:
            oob_count += 1
        
        features['area'][i] = clipped_size
        
        if clipped_size == 0:
            zero_area_count += 1
            continue
        
        if clipped_size < 3:
            small_roi_count += 1
        
        # Centroid
        w = lam / (lam.sum() + 1e-9)
        features['centroid_y'][i] = (ypix * w).sum()
        features['centroid_x'][i] = (xpix * w).sum()
        
        # Shape analysis (PCA)
        if clipped_size >= 3:
            coords = np.vstack([xpix, ypix]).T.astype(np.float32)
            c0 = coords - coords.mean(0)
            cov = (c0.T @ c0) / max(coords.shape[0] - 1, 1)
            evals, _ = np.linalg.eigh(cov)
            evals = np.sort(evals)[::-1]
            mj = np.sqrt(max(evals[0], 0.0))
            mn = np.sqrt(max(evals[1], 0.0))
        else:
            mj = mn = 0.0
        
        features['major_axis'][i] = mj
        features['minor_axis'][i] = mn
        features['elongation'][i] = mj / (mn + 1e-6)
        features['circ_ratio'][i] = mn / (mj + 1e-6)
        
        # Intensity features
        intens = A[ypix, xpix]
        features['inten_mean'][i] = np.nanmean(intens)
        features['inten_median'][i] = np.nanmedian(intens)
        features['inten_std'][i] = np.nanstd(intens) + 1e-9
        features['inten_contrast_z'][i] = (features['inten_mean'][i] - g_mean) / g_std
    
    data['roi_features'] = features
    
    # Summary statistics
    areas = features['area']
    elongations = features['elongation']
    
    print(f"Feature extraction complete:")
    print(f"  Out-of-bounds pixels: {oob_count}/{n_rois} ROIs affected")
    print(f"  Zero area ROIs: {zero_area_count}")
    print(f"  Small ROIs (<3 pixels): {small_roi_count}")
    print(f"  Area range: [{areas.min()}, {areas.max()}], median: {np.median(areas[areas>0]):.0f}")
    print(f"  Elongation range: [{elongations.min():.2f}, {elongations.max():.2f}]")

def classify_rois_simple(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """STEP 3C: Initial soma/process classification"""
    print("\n=== ROI CLASSIFICATION ===")
    
    feats = data.get('roi_features')
    if feats is None:
        raise ValueError("Features missing - run extract_all_roi_features first")
    
    lab_cfg = cfg['labeling']
    
    # Extract thresholds
    area_min = lab_cfg['soma_area_min']
    area_max = lab_cfg['soma_area_max']
    elong_max = lab_cfg['soma_elongation_max']
    circ_min = lab_cfg['soma_circularity_min']
    
    print(f"Classification criteria:")
    print(f"  Area: [{area_min}, {area_max}] pixels")
    print(f"  Elongation: <= {elong_max}")
    print(f"  Circularity: >= {circ_min}")
    
    # Apply criteria
    area = feats['area']
    elong = feats['elongation']
    circ = feats['circ_ratio']
    
    # Individual criteria
    area_pass = (area >= area_min) & (area <= area_max)
    elong_pass = (elong <= elong_max)
    circ_pass = (circ >= circ_min)
    
    # Combined
    soma_mask = area_pass & elong_pass & circ_pass
    
    print(f"Criteria breakdown:")
    print(f"  Area: {area_pass.sum()}/{len(area)} pass")
    print(f"  Elongation: {elong_pass.sum()}/{len(area)} pass")
    print(f"  Circularity: {circ_pass.sum()}/{len(area)} pass")
    print(f"  All criteria: {soma_mask.sum()}/{len(area)} pass")
    
    labels = np.where(soma_mask, 'soma', 'process').tolist()
    data['roi_labels'] = labels
    
    from collections import Counter
    counts = Counter(labels)
    print(f"Initial classification: {dict(counts)}")

def integrate_cellpose_iou(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """STEP 3D: Optional Cellpose integration"""
    print("\n=== CELLPOSE INTEGRATION ===")
    
    cp_cfg = cfg['cellpose']
    
    if not cp_cfg['enable']:
        print("Cellpose integration disabled in config")
        return
    
    print("Attempting to load Cellpose...")
    try:
        from cellpose import models
        print("Cellpose imported successfully")
    except Exception as e:
        print(f"Cellpose not available: {e}")
        return
    
    A = data.get('anatomy_image')
    if A is None:
        A = select_anatomical_image(data, cfg)
    
    model_type = cp_cfg['model']
    use_gpu = cp_cfg['gpu']
    diam = cp_cfg.get('diam')
    
    print(f"Cellpose settings:")
    print(f"  Model: {model_type}")
    print(f"  GPU: {use_gpu}")
    print(f"  Diameter: {diam}")
    print(f"  Image: {A.shape}")
    
    print("Running Cellpose segmentation...")
    model = models.Cellpose(model_type=model_type, gpu=use_gpu)
    masks, flows, styles, diams = model.eval(A, diameter=diam, channels=[0, 0])
    masks = np.array(masks)
    
    n_cellpose_objects = len(np.unique(masks)) - 1  # exclude background
    print(f"Cellpose found {n_cellpose_objects} objects")
    
    # Compute IoU for each Suite2p ROI
    h, w = A.shape
    ious = []
    high_iou_count = 0
    iou_threshold = cfg['labeling']['soma_iou_min']
    
    print("Computing IoU with Suite2p ROIs...")
    for i, st in enumerate(data['stat']):
        if i % 50 == 0 and i > 0:
            print(f"  IoU computed for {i}/{len(data['stat'])} ROIs...")
        
        ypix, xpix, _ = _roi_pixels_from_stat(st)
        ypix, xpix, _ = _clip_roi_pixels_to_image(ypix, xpix, _, h, w)
        
        if len(ypix) == 0:
            ious.append(0.0)
            continue
        
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[ypix, xpix] = 1
        
        best_iou = 0.0
        for m_id in np.unique(masks):
            if m_id == 0:  # skip background
                continue
            
            cp_mask = (masks == m_id)
            inter = np.logical_and(roi_mask, cp_mask).sum()
            if inter == 0:
                continue
            
            union = roi_mask.sum() + cp_mask.sum() - inter
            iou = inter / union
            best_iou = max(best_iou, iou)
        
        ious.append(best_iou)
        if best_iou >= iou_threshold:
            high_iou_count += 1
    
    data['cellpose_iou'] = np.array(ious, dtype=np.float32)
    
    print(f"IoU statistics:")
    print(f"  Mean IoU: {np.mean(ious):.3f}")
    print(f"  Max IoU: {np.max(ious):.3f}")
    print(f"  High IoU (>={iou_threshold}): {high_iou_count}/{len(ious)}")
    
    # Refine labels based on IoU
    if 'roi_labels' in data:
        labels = data['roi_labels']
        updated_count = 0
        
        for i, iou in enumerate(ious):
            if iou >= iou_threshold and labels[i] != 'soma':
                labels[i] = 'soma_cp'
                updated_count += 1
        
        print(f"Updated {updated_count} labels to 'soma_cp' based on Cellpose IoU")


# ==================== ENHANCED ROI FEATURE EXTRACTION ====================
def compute_roi_core_mask(ypix: np.ndarray, xpix: np.ndarray, lam: np.ndarray, percentile: float = 70.0) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ROI core pixels based on lambda percentile threshold - now using top 70%"""
    print(f"    Computing core mask with {percentile}th percentile threshold...")
    if lam.size == 0:
        return ypix, xpix
    
    lam_threshold = np.percentile(lam, percentile)
    core_mask = lam >= lam_threshold
    core_count = core_mask.sum()
    total_count = len(lam)
    
    print(f"      Core pixels: {core_count}/{total_count} ({100*core_count/total_count:.1f}%)")
    return ypix[core_mask], xpix[core_mask]

def compute_solidity(ypix: np.ndarray, xpix: np.ndarray, H: int, W: int) -> float:
    """Compute solidity = ROI_area / convex_hull_area"""
    if len(ypix) < 3:
        return 0.0
    
    roi_area = len(ypix)
    
    # Create convex hull
    from scipy.spatial import ConvexHull
    coords = np.column_stack([xpix, ypix])
    
    try:
        hull = ConvexHull(coords)
        hull_area = hull.volume  # 2D area
        solidity = roi_area / (hull_area + 1e-9)
        return min(1.0, solidity)  # Cap at 1.0 for numerical stability
    except:
        return 0.0

def compute_thickness_px(ypix: np.ndarray, xpix: np.ndarray, H: int, W: int) -> float:
    """Compute thickness as 2x median distance transform"""
    if len(ypix) < 3:
        return 0.0
    
    # Create binary mask
    mask = np.zeros((H, W), dtype=bool)
    mask[ypix, xpix] = True
    
    # Distance transform
    dist_transform = distance_transform_edt(mask)
    
    # Get distances at ROI pixels
    roi_distances = dist_transform[ypix, xpix]
    median_dist = np.median(roi_distances)
    
    return 2.0 * median_dist

def compute_orientation_coherence_refined(A: np.ndarray, ypix: np.ndarray, xpix: np.ndarray, H: int, W: int, dilation_radius: int = 2) -> float:
    """Compute orientation coherence from structure tensor with configurable dilation"""
    if len(ypix) < 5:
        print(f"      Too few pixels ({len(ypix)}) for coherence computation")
        return 0.0
    
    print(f"      Computing orientation coherence with dilation radius: {dilation_radius}")
    
    # Create dilated ROI mask
    mask = np.zeros((H, W), dtype=bool)
    mask[ypix, xpix] = True
    
    # Apply dilation
    for i in range(dilation_radius):
        mask = binary_dilation(mask)
    
    # Extract region from anatomy image with padding
    ymin, ymax = max(0, ypix.min()-3), min(H, ypix.max()+4)
    xmin, xmax = max(0, xpix.min()-3), min(W, xpix.max()+4)
    
    region = A[ymin:ymax, xmin:xmax]
    region_mask = mask[ymin:ymax, xmin:xmax]
    
    if region.size == 0 or not region_mask.any():
        print(f"      Empty region after extraction")
        return 0.0
    
    # Compute gradients
    grad_y = sobel(region, axis=0)
    grad_x = sobel(region, axis=1)
    
    # Structure tensor components
    Ixx = grad_x * grad_x
    Iyy = grad_y * grad_y
    Ixy = grad_x * grad_y
    
    # Average over the dilated region
    mask_indices = region_mask.nonzero()
    if len(mask_indices[0]) == 0:
        print(f"      No valid mask indices")
        return 0.0
    
    Sxx = np.mean(Ixx[mask_indices])
    Syy = np.mean(Iyy[mask_indices])
    Sxy = np.mean(Ixy[mask_indices])
    
    # Eigenvalues of structure tensor
    trace = Sxx + Syy
    det = Sxx * Syy - Sxy * Sxy
    
    if trace <= 1e-9:
        print(f"      Near-zero trace: {trace}")
        return 0.0
    
    discriminant = trace * trace - 4 * det
    if discriminant < 0:
        print(f"      Negative discriminant: {discriminant}")
        return 0.0
    
    lambda1 = 0.5 * (trace + np.sqrt(discriminant))
    lambda2 = 0.5 * (trace - np.sqrt(discriminant))
    
    # Coherence measure
    if lambda1 + lambda2 <= 1e-9:
        return 0.0
    
    coherence = (lambda1 - lambda2) / (lambda1 + lambda2)
    coherence_clamped = max(0.0, min(1.0, coherence))
    
    print(f"      Coherence: {coherence_clamped:.3f} (λ1={lambda1:.2e}, λ2={lambda2:.2e})")
    return coherence_clamped

def compute_skeleton_length(ypix: np.ndarray, xpix: np.ndarray, H: int, W: int) -> float:
    """Compute skeleton length for process-like structures"""
    if len(ypix) < 5:
        return 0.0
    
    # Create binary mask
    mask = np.zeros((H, W), dtype=bool)
    mask[ypix, xpix] = True
    
    # Skeletonize
    try:
        skeleton = skeletonize(mask)
        skeleton_length = np.sum(skeleton)
        return float(skeleton_length)
    except:
        return 0.0

def extract_refined_roi_features(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """STEP 3B: Extract refined ROI features with enhanced core extraction"""
    print("\n=== EXTRACTING REFINED ROI FEATURES ===")
    
    stat_list = data.get('stat')
    if stat_list is None:
        raise ValueError("stat list missing from data")
    
    n_stat = len(stat_list)
    n_rois = data['F'].shape[0]
    
    print(f"Processing {n_stat} stat entries for {n_rois} ROIs")
    
    if n_stat != n_rois:
        raise ValueError(f"stat length ({n_stat}) != F ROIs ({n_rois})")
    
    # Get anatomy image
    A = data.get('anatomy_image')
    if A is None:
        print("  Anatomy image not cached, selecting...")
        A = select_anatomical_image(data, cfg)
    else:
        print(f"  Using cached anatomy image: {A.shape}")
    
    # Global anatomy statistics
    g_mean = float(np.nanmean(A))
    g_std = float(np.nanstd(A) + 1e-9)
    print(f"Global anatomy stats: mean={g_mean:.1f}, std={g_std:.1f}")
    
    # Refined parameters
    core_percentile = 70.0  # Use top 70% instead of 30%
    dilation_radius = 2     # Configurable dilation for coherence
    print(f"Feature extraction parameters:")
    print(f"  Core percentile: {core_percentile}% (top pixels)")
    print(f"  Coherence dilation radius: {dilation_radius} pixels")
    
    # Initialize feature arrays
    features = {
        # Basic features
        'area': np.zeros(n_rois, dtype=np.int32),
        'area_core': np.zeros(n_rois, dtype=np.int32),
        'centroid_x': np.zeros(n_rois, dtype=np.float32),
        'centroid_y': np.zeros(n_rois, dtype=np.float32),
        
        # Shape features (enhanced)
        'major_axis': np.zeros(n_rois, dtype=np.float32),
        'minor_axis': np.zeros(n_rois, dtype=np.float32),
        'aspect_ratio': np.zeros(n_rois, dtype=np.float32),
        'elongation': np.zeros(n_rois, dtype=np.float32),    # legacy
        'circularity': np.zeros(n_rois, dtype=np.float32),
        'solidity': np.zeros(n_rois, dtype=np.float32),
        'thickness_px': np.zeros(n_rois, dtype=np.float32),
        
        # Process features (refined)
        'orientation_coherence': np.zeros(n_rois, dtype=np.float32),
        'skeleton_length': np.zeros(n_rois, dtype=np.float32),
        
        # Intensity features
        'inten_mean': np.zeros(n_rois, dtype=np.float32),
        'inten_median': np.zeros(n_rois, dtype=np.float32),
        'inten_std': np.zeros(n_rois, dtype=np.float32),
        'inten_contrast_z': np.zeros(n_rois, dtype=np.float32)
    }
    
    # Tracking variables
    oob_count = 0
    zero_area_count = 0
    small_roi_count = 0
    core_extraction_count = 0
    coherence_computation_count = 0
    
    print("  Processing ROIs with refined feature extraction...")
    for i, st in enumerate(stat_list):
        if i % 50 == 0 and i > 0:
            print(f"    Processed {i}/{n_rois} ROIs...")
        
        print(f"    ROI {i}: ", end="")
        
        ypix, xpix, lam = _roi_pixels_from_stat(st)
        original_size = ypix.size
        print(f"{original_size} pixels -> ", end="")
        
        # Bounds clipping
        ypix, xpix, lam = _clip_roi_pixels_to_image(ypix, xpix, lam, A.shape[0], A.shape[1])
        clipped_size = ypix.size
        print(f"{clipped_size} after clipping", end="")
        
        if clipped_size < original_size:
            oob_count += 1
            print(" (OOB)", end="")
        
        features['area'][i] = clipped_size
        
        if clipped_size == 0:
            zero_area_count += 1
            print(" -> ZERO AREA")
            continue
        
        if clipped_size < 3:
            small_roi_count += 1
            print(" -> TOO SMALL")
            continue
        
        print()  # New line for detailed processing
        
        # Extract core pixels for shape analysis (top 70%)
        ypix_core, xpix_core = compute_roi_core_mask(ypix, xpix, lam, core_percentile)
        features['area_core'][i] = len(ypix_core)
        
        if len(ypix_core) >= 3:
            core_extraction_count += 1
            shape_ypix, shape_xpix = ypix_core, xpix_core
            print(f"      Using core pixels for shape analysis: {len(ypix_core)}")
        else:
            shape_ypix, shape_xpix = ypix, xpix
            print(f"      Using all pixels for shape analysis: {len(ypix)}")
        
        # Centroid (using all pixels with lambda weighting)
        w = lam / (lam.sum() + 1e-9)
        features['centroid_y'][i] = (ypix * w).sum()
        features['centroid_x'][i] = (xpix * w).sum()
        print(f"      Centroid: ({features['centroid_x'][i]:.1f}, {features['centroid_y'][i]:.1f})")
        
        # Shape analysis (PCA on core pixels)
        if len(shape_ypix) >= 3:
            coords = np.vstack([shape_xpix, shape_ypix]).T.astype(np.float32)
            c0 = coords - coords.mean(0)
            cov = (c0.T @ c0) / max(coords.shape[0] - 1, 1)
            evals, _ = np.linalg.eigh(cov)
            evals = np.sort(evals)[::-1]
            mj = np.sqrt(max(evals[0], 0.0))
            mn = np.sqrt(max(evals[1], 0.0))
        else:
            mj = mn = 0.0
        
        features['major_axis'][i] = mj
        features['minor_axis'][i] = mn
        aspect_ratio = mj / (mn + 1e-6)
        features['aspect_ratio'][i] = aspect_ratio
        features['elongation'][i] = aspect_ratio  # Legacy compatibility
        
        print(f"      Shape: major={mj:.2f}, minor={mn:.2f}, AR={aspect_ratio:.2f}")
        
        # Enhanced shape features
        solidity = compute_solidity(shape_ypix, shape_xpix, A.shape[0], A.shape[1])
        features['solidity'][i] = solidity
        
        thickness = compute_thickness_px(ypix, xpix, A.shape[0], A.shape[1])
        features['thickness_px'][i] = thickness
        
        print(f"      Enhanced: solidity={solidity:.3f}, thickness={thickness:.2f}")
        
        # Circularity computation (4π*area/perimeter²)
        if len(shape_ypix) >= 3:
            # Create mask for perimeter estimation
            mask = np.zeros((A.shape[0], A.shape[1]), dtype=bool)
            mask[shape_ypix, shape_xpix] = True
            
            # Estimate perimeter from boundary pixels
            eroded = binary_erosion(mask, structure=np.ones((3,3)))
            boundary = mask & (~eroded)
            perimeter = np.sum(boundary)
            
            if perimeter > 0:
                circularity = 4 * np.pi * len(shape_ypix) / (perimeter * perimeter)
                features['circularity'][i] = min(1.0, circularity)
            else:
                features['circularity'][i] = 0.0
        else:
            features['circularity'][i] = 0.0
        
        print(f"      Circularity: {features['circularity'][i]:.3f}")
        
        # Process-specific features
        if len(ypix) >= 5:
            coherence = compute_orientation_coherence_refined(A, ypix, xpix, A.shape[0], A.shape[1], dilation_radius)
            features['orientation_coherence'][i] = coherence
            coherence_computation_count += 1
        else:
            features['orientation_coherence'][i] = 0.0
        
        skeleton_len = compute_skeleton_length(ypix, xpix, A.shape[0], A.shape[1])
        features['skeleton_length'][i] = skeleton_len
        
        print(f"      Process features: coherence={features['orientation_coherence'][i]:.3f}, skeleton_len={skeleton_len:.1f}")
        
        # Intensity features
        intens = A[ypix, xpix]
        features['inten_mean'][i] = np.nanmean(intens)
        features['inten_median'][i] = np.nanmedian(intens)
        features['inten_std'][i] = np.nanstd(intens) + 1e-9
        features['inten_contrast_z'][i] = (features['inten_mean'][i] - g_mean) / g_std
        
        print(f"      Intensity: mean={features['inten_mean'][i]:.1f}, contrast_z={features['inten_contrast_z'][i]:.2f}")
    
    data['roi_features'] = features
    
    # Summary statistics
    areas = features['area']
    aspect_ratios = features['aspect_ratio']
    solidities = features['solidity']
    circularities = features['circularity']
    thicknesses = features['thickness_px']
    coherences = features['orientation_coherence']
    skeleton_lens = features['skeleton_length']
    
    print(f"\nRefined feature extraction complete:")
    print(f"  Processing summary:")
    print(f"    Out-of-bounds pixels: {oob_count}/{n_rois} ROIs affected")
    print(f"    Zero area ROIs: {zero_area_count}")
    print(f"    Small ROIs (<3 pixels): {small_roi_count}")
    print(f"    Core extraction used: {core_extraction_count}/{n_rois} ROIs")
    print(f"    Coherence computed: {coherence_computation_count}/{n_rois} ROIs")
    
    print(f"  Feature ranges:")
    valid_mask = areas > 0
    print(f"    Area: [{areas[valid_mask].min()}, {areas[valid_mask].max()}], median: {np.median(areas[valid_mask]):.0f}")
    print(f"    Aspect ratio: [{aspect_ratios[valid_mask].min():.2f}, {aspect_ratios[valid_mask].max():.2f}], median: {np.median(aspect_ratios[valid_mask]):.2f}")
    print(f"    Solidity: [{solidities[valid_mask].min():.3f}, {solidities[valid_mask].max():.3f}], median: {np.median(solidities[valid_mask]):.3f}")
    print(f"    Circularity: [{circularities[valid_mask].min():.3f}, {circularities[valid_mask].max():.3f}], median: {np.median(circularities[valid_mask]):.3f}")
    print(f"    Thickness: [{thicknesses[valid_mask].min():.2f}, {thicknesses[valid_mask].max():.2f}], median: {np.median(thicknesses[valid_mask]):.2f}")
    print(f"    Coherence: [{coherences[valid_mask].min():.3f}, {coherences[valid_mask].max():.3f}], median: {np.median(coherences[valid_mask]):.3f}")
    print(f"    Skeleton length: [{skeleton_lens[valid_mask].min():.1f}, {skeleton_lens[valid_mask].max():.1f}], median: {np.median(skeleton_lens[valid_mask]):.1f}")


def extract_enhanced_roi_features(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """STEP 3B: Extract enhanced shape/intensity features for each ROI"""
    print("\n=== EXTRACTING ENHANCED ROI FEATURES ===")
    
    stat_list = data.get('stat')
    if stat_list is None:
        raise ValueError("stat list missing from data")
    
    n_stat = len(stat_list)
    n_rois = data['F'].shape[0]
    
    print(f"stat entries: {n_stat}, F ROIs: {n_rois}")
    
    if n_stat != n_rois:
        raise ValueError(f"stat length ({n_stat}) != F ROIs ({n_rois})")
    
    # Get anatomy image
    A = data.get('anatomy_image')
    if A is None:
        print("  Anatomy image not cached, selecting...")
        A = select_anatomical_image(data, cfg)
    else:
        print(f"  Using cached anatomy image: {A.shape}")
    
    # Global stats
    g_mean = float(np.nanmean(A))
    g_std = float(np.nanstd(A) + 1e-9)
    print(f"Global anatomy stats: mean={g_mean:.1f}, std={g_std:.1f}")
    
    # Core extraction parameters
    core_percentile = 30.0
    print(f"Using core percentile: {core_percentile}%")
    
    # Initialize feature arrays
    features = {
        # Basic features
        'area': np.zeros(n_rois, dtype=np.int32),
        'area_core': np.zeros(n_rois, dtype=np.int32),
        'centroid_x': np.zeros(n_rois, dtype=np.float32),
        'centroid_y': np.zeros(n_rois, dtype=np.float32),
        
        # Shape features (enhanced)
        'major_axis': np.zeros(n_rois, dtype=np.float32),
        'minor_axis': np.zeros(n_rois, dtype=np.float32),
        'aspect_ratio': np.zeros(n_rois, dtype=np.float32),  # major/minor
        'elongation': np.zeros(n_rois, dtype=np.float32),    # legacy (major/minor)
        'circularity': np.zeros(n_rois, dtype=np.float32),   # 4π*area/perimeter²
        'solidity': np.zeros(n_rois, dtype=np.float32),      # area/convex_hull_area
        'thickness_px': np.zeros(n_rois, dtype=np.float32),  # 2x median distance transform
        
        # Process features
        'orientation_coherence': np.zeros(n_rois, dtype=np.float32),
        'skeleton_length': np.zeros(n_rois, dtype=np.float32),
        
        # Intensity features
        'inten_mean': np.zeros(n_rois, dtype=np.float32),
        'inten_median': np.zeros(n_rois, dtype=np.float32),
        'inten_std': np.zeros(n_rois, dtype=np.float32),
        'inten_contrast_z': np.zeros(n_rois, dtype=np.float32)
    }
    
    oob_count = 0
    zero_area_count = 0
    small_roi_count = 0
    core_extraction_count = 0
    
    print("  Processing ROIs with enhanced features...")
    for i, st in enumerate(stat_list):
        if i % 100 == 0 and i > 0:
            print(f"    Processed {i}/{n_rois} ROIs...")
        
        ypix, xpix, lam = _roi_pixels_from_stat(st)
        original_size = ypix.size
        
        # Bounds clipping
        ypix, xpix, lam = _clip_roi_pixels_to_image(ypix, xpix, lam, A.shape[0], A.shape[1])
        clipped_size = ypix.size
        
        if clipped_size < original_size:
            oob_count += 1
        
        features['area'][i] = clipped_size
        
        if clipped_size == 0:
            zero_area_count += 1
            continue
        
        if clipped_size < 3:
            small_roi_count += 1
            continue
        
        # Extract core pixels for shape analysis
        ypix_core, xpix_core = compute_roi_core_mask(ypix, xpix, lam, core_percentile)
        features['area_core'][i] = len(ypix_core)
        
        if len(ypix_core) >= 3:
            core_extraction_count += 1
            shape_ypix, shape_xpix = ypix_core, xpix_core
        else:
            shape_ypix, shape_xpix = ypix, xpix
        
        # Centroid (using all pixels with lambda weighting)
        w = lam / (lam.sum() + 1e-9)
        features['centroid_y'][i] = (ypix * w).sum()
        features['centroid_x'][i] = (xpix * w).sum()
        
        # Shape analysis (PCA on core pixels)
        if len(shape_ypix) >= 3:
            coords = np.vstack([shape_xpix, shape_ypix]).T.astype(np.float32)
            c0 = coords - coords.mean(0)
            cov = (c0.T @ c0) / max(coords.shape[0] - 1, 1)
            evals, _ = np.linalg.eigh(cov)
            evals = np.sort(evals)[::-1]
            mj = np.sqrt(max(evals[0], 0.0))
            mn = np.sqrt(max(evals[1], 0.0))
        else:
            mj = mn = 0.0
        
        features['major_axis'][i] = mj
        features['minor_axis'][i] = mn
        features['aspect_ratio'][i] = mj / (mn + 1e-6)
        features['elongation'][i] = mj / (mn + 1e-6)  # Legacy compatibility
        
        # Enhanced shape features
        features['solidity'][i] = compute_solidity(shape_ypix, shape_xpix, A.shape[0], A.shape[1])
        features['thickness_px'][i] = compute_thickness_px(ypix, xpix, A.shape[0], A.shape[1])
        
        # Circularity (4π*area/perimeter²) - approximate with shape pixels
        if len(shape_ypix) >= 3:
            # Estimate perimeter from boundary pixels
            mask = np.zeros((A.shape[0], A.shape[1]), dtype=bool)
            mask[shape_ypix, shape_xpix] = True
            
            # Simple perimeter estimation
            from scipy.ndimage import binary_erosion
            eroded = binary_erosion(mask, structure=np.ones((3,3)))
            boundary = mask & (~eroded)
            perimeter = np.sum(boundary)
            
            if perimeter > 0:
                circularity = 4 * np.pi * len(shape_ypix) / (perimeter * perimeter)
                features['circularity'][i] = min(1.0, circularity)
            else:
                features['circularity'][i] = 0.0
        else:
            features['circularity'][i] = 0.0
        
        # Process-specific features
        features['orientation_coherence'][i] = compute_orientation_coherence(A, ypix, xpix, A.shape[0], A.shape[1])
        features['skeleton_length'][i] = compute_skeleton_length(ypix, xpix, A.shape[0], A.shape[1])
        
        # Intensity features
        intens = A[ypix, xpix]
        features['inten_mean'][i] = np.nanmean(intens)
        features['inten_median'][i] = np.nanmedian(intens)
        features['inten_std'][i] = np.nanstd(intens) + 1e-9
        features['inten_contrast_z'][i] = (features['inten_mean'][i] - g_mean) / g_std
    
    data['roi_features'] = features
    
    # Summary statistics
    areas = features['area']
    aspect_ratios = features['aspect_ratio']
    solidities = features['solidity']
    circularities = features['circularity']
    thicknesses = features['thickness_px']
    
    print(f"Enhanced feature extraction complete:")
    print(f"  Out-of-bounds pixels: {oob_count}/{n_rois} ROIs affected")
    print(f"  Zero area ROIs: {zero_area_count}")
    print(f"  Small ROIs (<3 pixels): {small_roi_count}")
    print(f"  Core extraction used: {core_extraction_count}/{n_rois} ROIs")
    print(f"  Area range: [{areas.min()}, {areas.max()}], median: {np.median(areas[areas>0]):.0f}")
    print(f"  Aspect ratio range: [{aspect_ratios.min():.2f}, {aspect_ratios.max():.2f}]")
    print(f"  Solidity range: [{solidities.min():.3f}, {solidities.max():.3f}]")
    print(f"  Circularity range: [{circularities.min():.3f}, {circularities.max():.3f}]")
    print(f"  Thickness range: [{thicknesses.min():.2f}, {thicknesses.max():.2f}]")

# ==================== ENHANCED ROI CLASSIFICATION ====================
def classify_rois_refined(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """STEP 3C: Refined soma/process classification with tightened rules"""
    print("\n=== REFINED ROI CLASSIFICATION ===")
    
    feats = data.get('roi_features')
    if feats is None:
        raise ValueError("Features missing - run extract_refined_roi_features first")
    
    print("Refined classification criteria:")
    
    # Tightened soma criteria (must meet ALL)
    soma_aspect_max = 1.9
    soma_solidity_min = 0.86
    soma_circularity_min = 0.62
    soma_thickness_min = 2.0
    soma_area_min = 30  # Raised from 25 to reduce tiny specks
    
    print(f"  SOMA (must meet ALL 5 criteria):")
    print(f"    aspect_ratio <= {soma_aspect_max}")
    print(f"    solidity >= {soma_solidity_min}")
    print(f"    circularity >= {soma_circularity_min}")
    print(f"    thickness_px >= {soma_thickness_min}")
    print(f"    area_px >= {soma_area_min} (raised to reduce tiny specks)")
    
    # Refined process criteria (if NOT soma, pass ANY of the shape criteria AND circularity constraint)
    proc_aspect_min = 2.1
    proc_coherence_min = 0.60  # Lowered slightly from 0.65
    proc_skeleton_min = 12.0   # Raised from 10.0
    proc_circularity_max = 0.80  # Raised from 0.75 to reject lumpy blobs
    
    print(f"  PROCESS (if NOT soma, pass ANY shape criterion AND circularity):")
    print(f"    Shape criteria (pass ANY):")
    print(f"      - aspect_ratio >= {proc_aspect_min}")
    print(f"      - orientation_coherence >= {proc_coherence_min}")
    print(f"      - skeleton_length >= {proc_skeleton_min}")
    print(f"    AND circularity <= {proc_circularity_max} (rejects lumpy blobs)")
    
    print(f"  UNCERTAIN: everything in the gap zone")
    
    # Extract features
    area = feats['area']
    aspect_ratio = feats['aspect_ratio']
    solidity = feats['solidity']
    circularity = feats['circularity']
    thickness = feats['thickness_px']
    orientation_coh = feats['orientation_coherence']
    skeleton_len = feats['skeleton_length']
    
    n_rois = len(area)
    print(f"\nEvaluating {n_rois} ROIs...")
    
    # Soma criteria evaluation
    print("  Evaluating soma criteria...")
    soma_aspect = aspect_ratio <= soma_aspect_max
    soma_solid = solidity >= soma_solidity_min
    soma_circ = circularity >= soma_circularity_min
    soma_thick = thickness >= soma_thickness_min
    soma_area_ok = area >= soma_area_min
    
    soma_all = soma_aspect & soma_solid & soma_circ & soma_thick & soma_area_ok
    
    print(f"    Individual soma criteria:")
    print(f"      Aspect ratio <= {soma_aspect_max}: {soma_aspect.sum()}/{n_rois} ({100*soma_aspect.sum()/n_rois:.1f}%)")
    print(f"      Solidity >= {soma_solidity_min}: {soma_solid.sum()}/{n_rois} ({100*soma_solid.sum()/n_rois:.1f}%)")
    print(f"      Circularity >= {soma_circularity_min}: {soma_circ.sum()}/{n_rois} ({100*soma_circ.sum()/n_rois:.1f}%)")
    print(f"      Thickness >= {soma_thickness_min}: {soma_thick.sum()}/{n_rois} ({100*soma_thick.sum()/n_rois:.1f}%)")
    print(f"      Area >= {soma_area_min}: {soma_area_ok.sum()}/{n_rois} ({100*soma_area_ok.sum()/n_rois:.1f}%)")
    print(f"    ALL soma criteria: {soma_all.sum()}/{n_rois} ({100*soma_all.sum()/n_rois:.1f}%)")
    
    # Process criteria evaluation (only for non-somas)
    print("  Evaluating process criteria...")
    proc_aspect = aspect_ratio >= proc_aspect_min
    proc_coherence = orientation_coh >= proc_coherence_min
    proc_skeleton = skeleton_len >= proc_skeleton_min
    proc_circ = circularity <= proc_circularity_max
    
    # Any of the shape criteria
    proc_shape_any = proc_aspect | proc_coherence | proc_skeleton
    # Combined process criteria
    proc_all = proc_shape_any & proc_circ
    
    print(f"    Individual process criteria:")
    print(f"      Aspect ratio >= {proc_aspect_min}: {proc_aspect.sum()}/{n_rois} ({100*proc_aspect.sum()/n_rois:.1f}%)")
    print(f"      Coherence >= {proc_coherence_min}: {proc_coherence.sum()}/{n_rois} ({100*proc_coherence.sum()/n_rois:.1f}%)")
    print(f"      Skeleton >= {proc_skeleton_min}: {proc_skeleton.sum()}/{n_rois} ({100*proc_skeleton.sum()/n_rois:.1f}%)")
    print(f"      ANY shape criterion: {proc_shape_any.sum()}/{n_rois} ({100*proc_shape_any.sum()/n_rois:.1f}%)")
    print(f"      Circularity <= {proc_circularity_max}: {proc_circ.sum()}/{n_rois} ({100*proc_circ.sum()/n_rois:.1f}%)")
    print(f"    ALL process criteria: {proc_all.sum()}/{n_rois} ({100*proc_all.sum()/n_rois:.1f}%)")
    
    # Handle Cellpose IoU if available
    iou_soma_force = np.zeros(n_rois, dtype=bool)
    iou_process_allow = np.ones(n_rois, dtype=bool)
    
    if 'cellpose_iou' in data:
        print("  Applying Cellpose IoU gates...")
        iou = data['cellpose_iou']
        iou_soma_force = iou >= 0.35
        iou_process_allow = iou <= 0.15
        print(f"    IoU >= 0.35 (force soma): {iou_soma_force.sum()}/{n_rois} ({100*iou_soma_force.sum()/n_rois:.1f}%)")
        print(f"    IoU <= 0.15 (allow process): {iou_process_allow.sum()}/{n_rois} ({100*iou_process_allow.sum()/n_rois:.1f}%)")
        print(f"    IoU in middle zone: {(~iou_soma_force & ~iou_process_allow).sum()}/{n_rois}")
    
    # Final classification logic
    print("  Applying final classification logic...")
    labels = []
    soma_count = 0
    process_count = 0
    uncertain_count = 0
    iou_forced_soma_count = 0
    
    for i in range(n_rois):
        if iou_soma_force[i]:
            # Force soma based on high IoU
            labels.append('soma')
            soma_count += 1
            iou_forced_soma_count += 1
        elif soma_all[i]:
            # Meets all soma criteria
            labels.append('soma')
            soma_count += 1
        elif proc_all[i] and iou_process_allow[i]:
            # Meets process criteria and IoU allows
            labels.append('process')
            process_count += 1
        else:
            # Uncertain cases
            labels.append('uncertain')
            uncertain_count += 1
    
    data['roi_labels'] = labels
    
    # Final classification summary
    print(f"\nFinal classification results:")
    total = len(labels)
    print(f"  Soma: {soma_count}/{total} ({100*soma_count/total:.1f}%)")
    if iou_forced_soma_count > 0:
        print(f"    (including {iou_forced_soma_count} IoU-forced)")
    print(f"  Process: {process_count}/{total} ({100*process_count/total:.1f}%)")
    print(f"  Uncertain: {uncertain_count}/{total} ({100*uncertain_count/total:.1f}%)")
    
    # QC warnings and recommendations
    uncertain_pct = 100 * uncertain_count / total
    if uncertain_pct > 15:
        print(f"\n*** WARNING: High uncertain fraction ({uncertain_pct:.1f}%) ***")
        print("    Recommendations:")
        print("    - Check if process criteria are too strict")
        print("    - Consider lowering coherence threshold or skeleton length")
        print("    - Raise circularity cap if many elongated somas are uncertain")
    elif uncertain_pct < 5:
        print(f"\n*** NOTE: Very low uncertain fraction ({uncertain_pct:.1f}%) ***")
        print("    This suggests good separation, but verify no edge cases are missed")
    else:
        print(f"\n*** Good uncertain fraction ({uncertain_pct:.1f}%) - suitable for manual review ***")
    
    # Feature distribution insights
    print(f"\nFeature distribution insights:")
    soma_mask = np.array(labels) == 'soma'
    proc_mask = np.array(labels) == 'process'
    unc_mask = np.array(labels) == 'uncertain'
    
    if soma_mask.sum() > 0:
        soma_ar_range = f"[{aspect_ratio[soma_mask].min():.2f}, {aspect_ratio[soma_mask].max():.2f}]"
        soma_thick_range = f"[{thickness[soma_mask].min():.2f}, {thickness[soma_mask].max():.2f}]"
        print(f"  Soma AR range: {soma_ar_range}, thickness range: {soma_thick_range}")
    
    if proc_mask.sum() > 0:
        proc_ar_range = f"[{aspect_ratio[proc_mask].min():.2f}, {aspect_ratio[proc_mask].max():.2f}]"
        proc_coh_range = f"[{orientation_coh[proc_mask].min():.3f}, {orientation_coh[proc_mask].max():.3f}]"
        print(f"  Process AR range: {proc_ar_range}, coherence range: {proc_coh_range}")
    
    if unc_mask.sum() > 0:
        unc_ar_range = f"[{aspect_ratio[unc_mask].min():.2f}, {aspect_ratio[unc_mask].max():.2f}]"
        print(f"  Uncertain AR range: {unc_ar_range} (check for gap zone)")



def classify_rois_enhanced(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """STEP 3C: Enhanced soma/process classification with refined rules"""
    print("\n=== ENHANCED ROI CLASSIFICATION ===")
    
    feats = data.get('roi_features')
    if feats is None:
        raise ValueError("Features missing - run extract_enhanced_roi_features first")
    
    # Enhanced classification criteria
    print("Enhanced classification criteria:")
    
    # Soma criteria (must meet ALL)
    soma_aspect_max = 1.9
    soma_solidity_min = 0.86
    soma_circularity_min = 0.62
    soma_thickness_min = 2.0
    soma_area_min = 25
    
    print(f"  SOMA (must meet ALL):")
    print(f"    aspect_ratio <= {soma_aspect_max}")
    print(f"    solidity >= {soma_solidity_min}")
    print(f"    circularity >= {soma_circularity_min}")
    print(f"    thickness_px >= {soma_thickness_min}")
    print(f"    area_px >= {soma_area_min}")
    
    # Process criteria (meet either A or B, plus circularity constraint)
    proc_aspect_min = 2.1
    proc_thickness_max = 1.8
    proc_coherence_min = 0.65
    proc_skeleton_min = 10.0
    proc_circularity_max = 0.75
    
    print(f"  PROCESS (meet either A or B, AND circularity constraint):")
    print(f"    Block A (shape): aspect_ratio >= {proc_aspect_min} OR thickness_px <= {proc_thickness_max}")
    print(f"    Block B (line-like): orientation_coherence >= {proc_coherence_min} OR skeleton_length >= {proc_skeleton_min}")
    print(f"    AND: circularity <= {proc_circularity_max}")
    
    # Extract features
    area = feats['area']
    aspect_ratio = feats['aspect_ratio']
    solidity = feats['solidity']
    circularity = feats['circularity']
    thickness = feats['thickness_px']
    orientation_coh = feats['orientation_coherence']
    skeleton_len = feats['skeleton_length']
    
    # Soma criteria evaluation
    soma_aspect = aspect_ratio <= soma_aspect_max
    soma_solid = solidity >= soma_solidity_min
    soma_circ = circularity >= soma_circularity_min
    soma_thick = thickness >= soma_thickness_min
    soma_area = area >= soma_area_min
    
    soma_all = soma_aspect & soma_solid & soma_circ & soma_thick & soma_area
    
    # Process criteria evaluation
    proc_shape_a = (aspect_ratio >= proc_aspect_min) | (thickness <= proc_thickness_max)
    proc_shape_b = (orientation_coh >= proc_coherence_min) | (skeleton_len >= proc_skeleton_min)
    proc_circ = circularity <= proc_circularity_max
    
    proc_all = (proc_shape_a | proc_shape_b) & proc_circ
    
    # Handle Cellpose IoU if available
    iou_soma_force = np.zeros(len(area), dtype=bool)
    iou_process_allow = np.ones(len(area), dtype=bool)
    
    if 'cellpose_iou' in data:
        print("  Applying Cellpose IoU gates:")
        iou = data['cellpose_iou']
        iou_soma_force = iou >= 0.35
        iou_process_allow = iou <= 0.15
        print(f"    IoU >= 0.35 (force soma): {iou_soma_force.sum()}")
        print(f"    IoU <= 0.15 (allow process): {iou_process_allow.sum()}")
    
    # Final classification logic
    labels = []
    soma_count = 0
    process_count = 0
    uncertain_count = 0
    
    for i in range(len(area)):
        if iou_soma_force[i]:
            # Force soma based on high IoU
            labels.append('soma')
            soma_count += 1
        elif soma_all[i]:
            # Meets all soma criteria
            labels.append('soma')
            soma_count += 1
        elif proc_all[i] and iou_process_allow[i]:
            # Meets process criteria and IoU allows
            labels.append('process')
            process_count += 1
        else:
            # Uncertain cases
            labels.append('uncertain')
            uncertain_count += 1
    
    data['roi_labels'] = labels
    
    # Detailed breakdown
    print(f"Criteria breakdown:")
    print(f"  Soma aspect_ratio: {soma_aspect.sum()}/{len(area)} pass")
    print(f"  Soma solidity: {soma_solid.sum()}/{len(area)} pass")
    print(f"  Soma circularity: {soma_circ.sum()}/{len(area)} pass")
    print(f"  Soma thickness: {soma_thick.sum()}/{len(area)} pass")
    print(f"  Soma area: {soma_area.sum()}/{len(area)} pass")
    print(f"  ALL soma criteria: {soma_all.sum()}/{len(area)} pass")
    print()
    print(f"  Process shape A: {proc_shape_a.sum()}/{len(area)} pass")
    print(f"  Process shape B: {proc_shape_b.sum()}/{len(area)} pass")
    print(f"  Process circularity: {proc_circ.sum()}/{len(area)} pass")
    print(f"  Process criteria: {proc_all.sum()}/{len(area)} pass")
    
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    
    print(f"\nFinal classification:")
    for label, count in counts.items():
        pct = 100 * count / total
        print(f"  {label}: {count}/{total} ({pct:.1f}%)")
    
    # QC warnings
    uncertain_pct = 100 * uncertain_count / total
    if uncertain_pct > 15:
        print(f"\n*** WARNING: High uncertain fraction ({uncertain_pct:.1f}%) ***")
        print("    Consider widening process criteria or raising soma solidity threshold")

# ==================== QC VISUALIZATION ====================

def plot_refined_feature_distributions(data: Dict[str, Any], cfg: Dict[str, Any], save: bool = True):
    """Plot refined feature distributions with uncertainty class"""
    print("\n=== PLOTTING REFINED FEATURE DISTRIBUTIONS ===")
    
    feats = data.get('roi_features')
    labels = data.get('roi_labels')
    
    if feats is None or labels is None:
        raise ValueError("Features and labels required")
    
    aspect_ratio = feats['aspect_ratio']
    thickness = feats['thickness_px']
    solidity = feats['solidity']
    circularity = feats['circularity']
    coherence = feats['orientation_coherence']
    skeleton_len = feats['skeleton_length']
    
    # Filter out invalid values
    valid_mask = (aspect_ratio > 0) & (thickness > 0) & np.isfinite(aspect_ratio) & np.isfinite(thickness)
    
    print(f"Valid ROIs for plotting: {valid_mask.sum()}/{len(valid_mask)}")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Color mapping including uncertain
    label_colors = {'soma': 'red', 'process': 'blue', 'uncertain': 'orange'}
    label_order = ['soma', 'process', 'uncertain']
    
    # 1. Aspect ratio histogram
    ax = axes[0, 0]
    for label in label_order:
        mask = valid_mask & (np.array(labels) == label)
        if mask.sum() > 0:
            ax.hist(aspect_ratio[mask], bins=50, alpha=0.6, 
                   color=label_colors[label], label=f"{label} (n={mask.sum()})")
    ax.axvline(1.9, color='red', linestyle='--', alpha=0.7, label='soma cutoff')
    ax.axvline(2.1, color='blue', linestyle='--', alpha=0.7, label='process cutoff')
    ax.set_xlabel('Aspect Ratio')
    ax.set_ylabel('Count')
    ax.set_title('Aspect Ratio Distribution')
    ax.legend(fontsize=8)
    
    # 2. Thickness histogram
    ax = axes[0, 1]
    for label in label_order:
        mask = valid_mask & (np.array(labels) == label)
        if mask.sum() > 0:
            ax.hist(thickness[mask], bins=50, alpha=0.6, 
                   color=label_colors[label], label=f"{label} (n={mask.sum()})")
    ax.axvline(2.0, color='red', linestyle='--', alpha=0.7, label='soma cutoff')
    ax.set_xlabel('Thickness (px)')
    ax.set_ylabel('Count')
    ax.set_title('Thickness Distribution')
    ax.legend(fontsize=8)
    
    # 3. Solidity histogram
    ax = axes[0, 2]
    for label in label_order:
        mask = valid_mask & (np.array(labels) == label)
        if mask.sum() > 0:
            ax.hist(solidity[mask], bins=50, alpha=0.6, 
                   color=label_colors[label], label=f"{label} (n={mask.sum()})")
    ax.axvline(0.86, color='red', linestyle='--', alpha=0.7, label='soma cutoff')
    ax.set_xlabel('Solidity')
    ax.set_ylabel('Count')
    ax.set_title('Solidity Distribution')
    ax.legend(fontsize=8)
    
    # 4. Circularity histogram
    ax = axes[1, 0]
    for label in label_order:
        mask = valid_mask & (np.array(labels) == label)
        if mask.sum() > 0:
            ax.hist(circularity[mask], bins=50, alpha=0.6, 
                   color=label_colors[label], label=f"{label} (n={mask.sum()})")
    ax.axvline(0.62, color='red', linestyle='--', alpha=0.7, label='soma cutoff')
    ax.axvline(0.80, color='blue', linestyle='--', alpha=0.7, label='process cutoff')
    ax.set_xlabel('Circularity')
    ax.set_ylabel('Count')
    ax.set_title('Circularity Distribution')
    ax.legend(fontsize=8)
    
    # 5. Orientation coherence histogram
    ax = axes[1, 1]
    for label in label_order:
        mask = valid_mask & (np.array(labels) == label)
        if mask.sum() > 0:
            ax.hist(coherence[mask], bins=50, alpha=0.6, 
                   color=label_colors[label], label=f"{label} (n={mask.sum()})")
    ax.axvline(0.60, color='blue', linestyle='--', alpha=0.7, label='process cutoff')
    ax.set_xlabel('Orientation Coherence')
    ax.set_ylabel('Count')
    ax.set_title('Orientation Coherence Distribution')
    ax.legend(fontsize=8)
    
    # 6. Skeleton length histogram
    ax = axes[1, 2]
    for label in label_order:
        mask = valid_mask & (np.array(labels) == label)
        if mask.sum() > 0:
            ax.hist(skeleton_len[mask], bins=50, alpha=0.6, 
                   color=label_colors[label], label=f"{label} (n={mask.sum()})")
    ax.axvline(12.0, color='blue', linestyle='--', alpha=0.7, label='process cutoff')
    ax.set_xlabel('Skeleton Length (px)')
    ax.set_ylabel('Count')
    ax.set_title('Skeleton Length Distribution')
    ax.legend(fontsize=8)
    
    # 7. 2D scatter: Aspect ratio vs Thickness
    ax = axes[2, 0]
    for label in label_order:
        mask = valid_mask & (np.array(labels) == label)
        if mask.sum() > 0:
            ax.scatter(aspect_ratio[mask], thickness[mask], 
                      c=label_colors[label], alpha=0.6, s=15, label=label)
    
    ax.axvline(1.9, color='red', linestyle='--', alpha=0.5)
    ax.axvline(2.1, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(2.0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Aspect Ratio')
    ax.set_ylabel('Thickness (px)')
    ax.set_title('Aspect Ratio vs Thickness\n(Look for valley at cutoffs)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, min(15, np.percentile(aspect_ratio[valid_mask], 99)))
    ax.set_ylim(0, min(10, np.percentile(thickness[valid_mask], 99)))
    
    # 8. 2D scatter: Aspect ratio vs Coherence
    ax = axes[2, 1]
    for label in label_order:
        mask = valid_mask & (np.array(labels) == label)
        if mask.sum() > 0:
            ax.scatter(aspect_ratio[mask], coherence[mask], 
                      c=label_colors[label], alpha=0.6, s=15, label=label)
    
    ax.axvline(2.1, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(0.60, color='blue', linestyle='--', alpha=0.5)
    ax.set_xlabel('Aspect Ratio')
    ax.set_ylabel('Orientation Coherence')
    ax.set_title('Aspect Ratio vs Coherence\n(Process criteria visualization)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, min(15, np.percentile(aspect_ratio[valid_mask], 99)))
    
    # 9. Feature correlation matrix
    ax = axes[2, 2]
    feature_names = ['aspect_ratio', 'thickness_px', 'solidity', 'circularity', 
                    'orientation_coherence', 'skeleton_length']
    feature_data = np.column_stack([feats[name][valid_mask] for name in feature_names])
    
    corr_matrix = np.corrcoef(feature_data.T)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels([name.replace('_', '\n') for name in feature_names], fontsize=8)
    ax.set_yticklabels([name.replace('_', '\n') for name in feature_names], fontsize=8)
    ax.set_title('Feature Correlations')
    
    # Add correlation values
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            color = 'white' if abs(corr_matrix[i,j]) > 0.5 else 'black'
            ax.text(j, i, f'{corr_matrix[i,j]:.2f}', 
                   ha='center', va='center', color=color, fontsize=8)
    
    plt.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    
    if save:
        save_dir = cfg['overlay']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'refined_feature_distributions.png')
        fig.savefig(save_path, dpi=cfg['overlay']['dpi'], bbox_inches='tight')
        print(f"Refined feature distributions saved: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\nFeature distribution summary:")
    for label in label_order:
        mask = valid_mask & (np.array(labels) == label)
        if mask.sum() > 0:
            print(f"  {label.upper()} (n={mask.sum()}):")
            print(f"    Aspect ratio: {aspect_ratio[mask].mean():.2f} ± {aspect_ratio[mask].std():.2f}")
            print(f"    Thickness: {thickness[mask].mean():.2f} ± {thickness[mask].std():.2f}")
            print(f"    Coherence: {coherence[mask].mean():.3f} ± {coherence[mask].std():.3f}")
            print(f"    Circularity: {circularity[mask].mean():.3f} ± {circularity[mask].std():.3f}")
    
    print("Refined feature distribution plots complete")

def plot_feature_distributions(data: Dict[str, Any], cfg: Dict[str, Any], save: bool = True):
    """Plot feature distributions for QC"""
    print("\n=== PLOTTING FEATURE DISTRIBUTIONS ===")
    
    feats = data.get('roi_features')
    labels = data.get('roi_labels')
    
    if feats is None or labels is None:
        raise ValueError("Features and labels required")
    
    aspect_ratio = feats['aspect_ratio']
    thickness = feats['thickness_px']
    solidity = feats['solidity']
    circularity = feats['circularity']
    
    # Filter out invalid values
    valid_mask = (aspect_ratio > 0) & (thickness > 0) & np.isfinite(aspect_ratio) & np.isfinite(thickness)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Aspect ratio histogram
    ax = axes[0, 0]
    for label in ['soma', 'process', 'uncertain']:
        mask = valid_mask & (np.array(labels) == label)
        if mask.sum() > 0:
            ax.hist(aspect_ratio[mask], bins=50, alpha=0.6, label=f"{label} (n={mask.sum()})")
    ax.axvline(1.9, color='red', linestyle='--', alpha=0.7, label='soma cutoff')
    ax.axvline(2.1, color='blue', linestyle='--', alpha=0.7, label='process cutoff')
    ax.set_xlabel('Aspect Ratio')
    ax.set_ylabel('Count')
    ax.set_title('Aspect Ratio Distribution')
    ax.legend()
    
    # Thickness histogram
    ax = axes[0, 1]
    for label in ['soma', 'process', 'uncertain']:
        mask = valid_mask & (np.array(labels) == label)
        if mask.sum() > 0:
            ax.hist(thickness[mask], bins=50, alpha=0.6, label=f"{label} (n={mask.sum()})")
    ax.axvline(2.0, color='red', linestyle='--', alpha=0.7, label='soma cutoff')
    ax.axvline(1.8, color='blue', linestyle='--', alpha=0.7, label='process cutoff')
    ax.set_xlabel('Thickness (px)')
    ax.set_ylabel('Count')
    ax.set_title('Thickness Distribution')
    ax.legend()
    
    # Solidity histogram
    ax = axes[0, 2]
    for label in ['soma', 'process', 'uncertain']:
        mask = valid_mask & (np.array(labels) == label)
        if mask.sum() > 0:
            ax.hist(solidity[mask], bins=50, alpha=0.6, label=f"{label} (n={mask.sum()})")
    ax.axvline(0.86, color='red', linestyle='--', alpha=0.7, label='soma cutoff')
    ax.set_xlabel('Solidity')
    ax.set_ylabel('Count')
    ax.set_title('Solidity Distribution')
    ax.legend()
    
    # Circularity histogram
    ax = axes[1, 0]
    for label in ['soma', 'process', 'uncertain']:
        mask = valid_mask & (np.array(labels) == label)
        if mask.sum() > 0:
            ax.hist(circularity[mask], bins=50, alpha=0.6, label=f"{label} (n={mask.sum()})")
    ax.axvline(0.62, color='red', linestyle='--', alpha=0.7, label='soma cutoff')
    ax.axvline(0.75, color='blue', linestyle='--', alpha=0.7, label='process cutoff')
    ax.set_xlabel('Circularity')
    ax.set_ylabel('Count')
    ax.set_title('Circularity Distribution')
    ax.legend()
    
    # 2D hexbin plot (aspect_ratio vs thickness)
    ax = axes[1, 1]
    
    # Create color map for labels
    label_colors = {'soma': 'red', 'process': 'blue', 'uncertain': 'orange'}
    
    for label in ['soma', 'process', 'uncertain']:
        mask = valid_mask & (np.array(labels) == label)
        if mask.sum() > 0:
            ax.scatter(aspect_ratio[mask], thickness[mask], 
                      c=label_colors[label], alpha=0.6, s=10, label=label)
    
    ax.axvline(1.9, color='red', linestyle='--', alpha=0.5)
    ax.axvline(2.1, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(2.0, color='red', linestyle='--', alpha=0.5)
    ax.axhline(1.8, color='blue', linestyle='--', alpha=0.5)
    ax.set_xlabel('Aspect Ratio')
    ax.set_ylabel('Thickness (px)')
    ax.set_title('Aspect Ratio vs Thickness')
    ax.legend()
    ax.set_xlim(0, min(10, np.percentile(aspect_ratio[valid_mask], 99)))
    ax.set_ylim(0, min(10, np.percentile(thickness[valid_mask], 99)))
    
    # Feature correlation matrix
    ax = axes[1, 2]
    feature_names = ['aspect_ratio', 'thickness_px', 'solidity', 'circularity']
    feature_data = np.column_stack([feats[name][valid_mask] for name in feature_names])
    
    corr_matrix = np.corrcoef(feature_data.T)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45)
    ax.set_yticklabels(feature_names)
    ax.set_title('Feature Correlations')
    
    # Add correlation values
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            ax.text(j, i, f'{corr_matrix[i,j]:.2f}', 
                   ha='center', va='center', color='white' if abs(corr_matrix[i,j]) > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    if save:
        save_dir = cfg['overlay']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'feature_distributions.png')
        fig.savefig(save_path, dpi=cfg['overlay']['dpi'], bbox_inches='tight')
        print(f"Feature distributions saved: {save_path}")
    
    plt.show()
    
    print("Feature distribution plots complete")


# ==================== FLUORESCENCE PROCESSING ====================
def neuropil_regress_debug(data: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """STEP 4A: Neuropil correction"""
    print("\n=== NEUROPIL CORRECTION ===")
    
    np_cfg = cfg["neuropil"]
    F, Fneu = data["F"], data["Fneu"]
    N, T = F.shape
    
    print(f"Input: {N} ROIs x {T} timepoints")
    
    # if not np_cfg["enable"]:
    #     print("Using fixed alpha correction...")
    #     alpha = np_cfg["fallback_alpha"]
    #     print(f"  Fixed alpha: {alpha}")
                
    #     data["Fc"] = F - alpha * Fneu
    #     data["neuropil_a"] = np.full(N, alpha, dtype=np.float32)
        
    #     print(f"Fixed correction applied: Fc = F - {alpha} x Fneu")
    #     return data
    
    print("Using regression-based correction...")
    lam = float(np_cfg["lam"])
    lo, hi = np_cfg["bounds"]
    print(f"  Ridge parameter: {lam}")
    print(f"  Alpha bounds: [{lo}, {hi}]")
    
    ones = np.ones(T)
    a = np.zeros(N, dtype=np.float32)
    Fc = np.empty_like(F, dtype=np.float32)
    
    clipped_count = 0
    
    for i in range(N):
        if i % 100 == 0 and i > 0:
            print(f"    Processed {i}/{N} ROIs...")
        
        x = Fneu[i]
        X = np.stack([x, ones], axis=1)
        XtX = X.T @ X + lam * np.eye(2)
        beta = np.linalg.solve(XtX, X.T @ F[i])
        
        alpha_raw = float(beta[0])
        alpha_clipped = float(np.clip(alpha_raw, lo, hi))
        bias = float(beta[1])
        
        if alpha_raw != alpha_clipped:
            clipped_count += 1
        
        
        # Cap np slope to 0.7
        alpha_clipped = min(alpha_clipped, 0.7)
        a[i] = alpha_clipped
        # Fc[i] = F[i] - alpha_clipped * x - bias
        # use slope only for neuropil correction
        Fc[i] = F[i] - alpha_clipped * x
    
    data["Fc"] = Fc
    data["neuropil_a"] = a
    
    print(f"Regression correction complete:")
    print(f"  Alpha range: [{a.min():.3f}, {a.max():.3f}]")
    print(f"  Alpha mean +/- std: {a.mean():.3f} +/- {a.std():.3f}")
    print(f"  Clipped alphas: {clipped_count}/{N}")
    
    return data








def neuropil_regress(data: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """STEP 4A: Neuropil correction"""
    print("\n=== NEUROPIL CORRECTION ===")
    
    np_cfg = cfg["neuropil"]
    F, Fneu = data["F"], data["Fneu"]
    N, T = F.shape
    
    print(f"Input: {N} ROIs x {T} timepoints")
    
    if not np_cfg["enable"]:
        print("Using fixed alpha correction...")
        alpha = np_cfg["fallback_alpha"]
        print(f"  Fixed alpha: {alpha}")
                
        data["Fc"] = F - alpha * Fneu
        data["neuropil_a"] = np.full(N, alpha, dtype=np.float32)
        
        print(f"Fixed correction applied: Fc = F - {alpha} x Fneu")
        return data
    
    print("Using regression-based correction...")
    lam = float(np_cfg["lam"])
    lo, hi = np_cfg["bounds"]
    print(f"  Ridge parameter: {lam}")
    print(f"  Alpha bounds: [{lo}, {hi}]")
    
    ones = np.ones(T)
    a = np.zeros(N, dtype=np.float32)
    Fc = np.empty_like(F, dtype=np.float32)
    
    clipped_count = 0
    
    for i in range(N):
        if i % 100 == 0 and i > 0:
            print(f"    Processed {i}/{N} ROIs...")
        
        x = Fneu[i]
        X = np.stack([x, ones], axis=1)
        XtX = X.T @ X + lam * np.eye(2)
        beta = np.linalg.solve(XtX, X.T @ F[i])
        
        alpha_raw = float(beta[0])
        alpha_clipped = float(np.clip(alpha_raw, lo, hi))
        bias = float(beta[1])
        
        if alpha_raw != alpha_clipped:
            clipped_count += 1
        
        
        # Cap np slope to 0.7
        alpha_clipped = min(alpha_clipped, 0.7)
        a[i] = alpha_clipped
        # Fc[i] = F[i] - alpha_clipped * x - bias
        # use slope only for neuropil correction
        Fc[i] = F[i] - alpha_clipped * x
    
    data["Fc"] = Fc
    data["neuropil_a"] = a
    
    print(f"Regression correction complete:")
    print(f"  Alpha range: [{a.min():.3f}, {a.max():.3f}]")
    print(f"  Alpha mean +/- std: {a.mean():.3f} +/- {a.std():.3f}")
    print(f"  Clipped alphas: {clipped_count}/{N}")
    
    return data


def compute_percentile_baseline(F: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """Compute F0 baseline using percentile windowing (your current sophisticated method)"""
    print("  Computing percentile-based baseline...")
    
    fs = float(cfg["acq"]["fs"])
    base_cfg = cfg["baseline"]["custom"]
    
    win_s = base_cfg["win_s"]
    prc = base_cfg["percentile"]
    sig_s = base_cfg["smooth_sigma_s"]
    div = base_cfg["smooth_sigma_divisor"]
    
    print(f"    Window: {win_s}s ({int(win_s * fs)} samples)")
    print(f"    Percentile: {prc}")
    print(f"    Smoothing sigma: {sig_s}s")
    
    win = max(1, int(win_s * fs))
    N, T = F.shape
    
    # Create windows
    edges = np.arange(0, T, win)
    if edges[-1] != T:
        edges = np.append(edges, T)
    
    n_windows = len(edges) - 1
    print(f"    Created {n_windows} windows for baseline estimation")
    
    F0 = np.empty_like(F, dtype=np.float32)
    
    for i in range(N):
        if i % 100 == 0 and i > 0:
            print(f"      Processed {i}/{N} ROIs...")
        
        centers, vals = [], []
        for j in range(len(edges) - 1):
            s, e = edges[j], edges[j + 1]
            seg = F[i, s:e]
            if seg.size == 0:
                continue
            centers.append(0.5 * (s + e - 1))
            vals.append(np.nanpercentile(seg, prc))
        
        if len(vals) == 0:
            F0[i] = 0
        elif len(vals) == 1:
            F0[i] = vals[0]
        else:
            F0[i] = np.interp(np.arange(T), centers, vals).astype(np.float32)
        
        # Smoothing
        if sig_s > 0:
            sigma = (sig_s * fs) / div
            F0[i] = gaussian_filter(F0[i], sigma=sigma)
    
    print(f"    Percentile baseline complete: F0 range [{F0.min():.1f}, {F0.max():.1f}]")
    return F0


def compute_suite2p_baseline(data: Dict[str, Any], cfg: Dict[str, Any]) -> np.ndarray:
    """Compute F0 baseline using Suite2p's preprocess function"""
    print("  Computing Suite2p baseline...")
    
    from suite2p.extraction.dcnv import preprocess
    
    Fc = data["Fc"]
    fs = float(cfg["acq"]["fs"])
    base_cfg = cfg["baseline"]["suite2p"]
    
    method = base_cfg["method"]
    win_baseline = base_cfg["win_baseline"]
    sig_baseline = base_cfg["sig_baseline"]
    prctile_baseline = base_cfg.get("prctile_baseline", 8.0)
    
    print(f"    Method: {method}")
    print(f"    Window: {win_baseline}s")
    print(f"    Gaussian sigma: {sig_baseline} frames")
    if method == "constant_prctile":
        print(f"    Percentile: {prctile_baseline}")
    
    # Suite2p preprocess returns baseline-subtracted signal
    # We need to recover the baseline (F0) for consistency
    F_corrected = preprocess(
        F=Fc,
        baseline=method,
        win_baseline=win_baseline,
        sig_baseline=sig_baseline,
        fs=fs,
        prctile_baseline=prctile_baseline
    )
    
    # Recover baseline: F0 = Fc - F_corrected
    F0 = Fc - F_corrected
    
    print(f"    Suite2p baseline complete: F0 range [{F0.min():.1f}, {F0.max():.1f}]")
    return F0


def deconv_oasis_from_Fc_recon(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """
    OASIS deconvolution with full component reconstruction using individual traces.
    Stores: spks_oasis_recon (spikes), C_oasis_recon (denoised), oasis_b_recon, 
            oasis_g_recon, oasis_lam_recon for comparison with batch version.
    """
    print("  OASIS reconstruction deconvolution setup...")
    
    # Get configuration parameters
    deconv_cfg = cfg.get("deconvolution", {})
    tau_s = deconv_cfg.get("tau_s", 0.7)
    detrend_with_f0 = deconv_cfg.get("detrend_with_f0", True)
    
    fs = float(cfg["acq"]["fs"])
    Fc = data["Fc"]
    
    if Fc is None:
        raise ValueError("Fc required for reconstruction deconvolution")
    
    print(f"  Reconstruction deconvolution parameters:")
    print(f"    Sampling rate: {fs:.1f} Hz")
    print(f"    Tau (decay time): {tau_s}s")
    print(f"    Detrend with F0: {detrend_with_f0}")
    
    # Prepare fluorescence data
    F = Fc.astype(np.float32)
    N, T = F.shape
    
    # Optional gentle detrending with F0
    if detrend_with_f0 and "F0" in data:
        print("  Applying gentle F0 detrending...")
        F0 = data["F0"]
        median_f0 = np.median(F0, axis=1, keepdims=True)
        F = F - F0 + median_f0
        print(f"    Detrended signal range: [{F.min():.1f}, {F.max():.1f}]")
    else:
        print("  Using raw Fc (no F0 detrending)")
    
    print(f"  Processing {N} ROIs x {T} timepoints with trace reconstruction...")
    
    try:
        # Import the individual OASIS function for full outputs
        from suite2p.extraction.dcnv import oasis_trace
        
        # Calculate decay coefficient
        g_fixed = np.exp(-1.0 / (tau_s * fs))
        print(f"    Using decay coefficient: g = {g_fixed:.4f}")
        
        # Initialize output arrays
        C = np.zeros_like(F, dtype=np.float32)
        S = np.zeros_like(F, dtype=np.float32)
        b_list = []
        g_list = []
        lam_list = []
        
        deconv_errors = 0
        
        for i in range(N):
            if i % 50 == 0 and i > 0:
                print(f"    Reconstructed {i}/{N} ROIs...")
            
            try:
                # Prepare working arrays for oasis_trace
                v = np.zeros(T, dtype=np.float32)
                w = np.zeros(T, dtype=np.float32) 
                t = np.zeros(T, dtype=np.int64)
                l = np.zeros(T, dtype=np.float32)
                s = np.zeros(T, dtype=np.float32)
                
                # Call individual OASIS trace function
                # This modifies v, w, t, l, s in place
                oasis_trace(F[i], v, w, t, l, s, tau_s, fs)
                
                # Extract spikes
                S[i] = s.copy()
                
                # Reconstruct calcium signal from spikes
                # C[t] = sum of exponentially decaying spikes
                c_recon = np.zeros(T, dtype=np.float32)
                for j in range(T):
                    if s[j] > 0:
                        # Add exponentially decaying kernel
                        decay_length = min(T-j, int(5 * tau_s * fs))  # 5 time constants
                        decay_kernel = np.exp(-np.arange(decay_length) / (tau_s * fs))
                        c_recon[j:j+decay_length] += s[j] * decay_kernel
                
                C[i] = c_recon
                
                # Extract baseline (median of residual)
                baseline = np.median(F[i] - c_recon)
                b_list.append(float(baseline))
                g_list.append(g_fixed)
                lam_list.append(0.0)  # Not used in this version
                
            except Exception as e:
                print(f"    Warning: Reconstruction failed for ROI {i}: {e}")
                deconv_errors += 1
                # Fill with fallback values
                C[i] = F[i]  # Use original signal
                S[i] = 0
                b_list.append(0.0)
                g_list.append(g_fixed)
                lam_list.append(0.0)
        
        # Store reconstruction results with suffix
        data["C_oasis_recon"] = C
        data["spks_oasis_recon"] = S
        data["oasis_b_recon"] = np.array(b_list, dtype=np.float32)
        data["oasis_g_recon"] = np.array(g_list, dtype=np.float32)
        data["oasis_lam_recon"] = np.array(lam_list, dtype=np.float32)
        
        # Validation and summary
        if deconv_errors > 0:
            print(f"    WARNING: Reconstruction failed for {deconv_errors}/{N} ROIs")
        
        # Compute statistics
        total_spikes = np.sum(S > 0)
        mean_spike_rate = total_spikes / (N * T / fs)
        max_spike = np.max(S)
        
        print(f"  Reconstruction deconvolution results:")
        print(f"    Total spikes detected: {total_spikes}")
        print(f"    Mean spike rate: {mean_spike_rate:.3f} spikes/ROI/s")
        print(f"    Max spike amplitude: {max_spike:.3f}")
        print(f"    Mean decay coefficient: {np.mean(g_list):.4f}")
        print(f"    Mean baseline: {np.mean(b_list):.1f}")
        
        # Quality check
        if np.any(np.isnan(C)) or np.any(np.isnan(S)):
            print("    WARNING: NaN values detected in reconstruction results")
        
        # Denoising assessment
        residual_std = np.std(F - C)
        signal_std = np.std(C)
        snr_improvement = signal_std / (residual_std + 1e-9)
        print(f"    Denoising SNR improvement: {snr_improvement:.2f}x")
        
        print("  OASIS reconstruction deconvolution complete")
        
    except ImportError as e:
        print(f"  ERROR: Cannot import OASIS trace function: {e}")
        print("  Skipping reconstruction deconvolution...")
        return
    except Exception as e:
        print(f"  ERROR: OASIS reconstruction failed: {e}")
        print("  Skipping reconstruction deconvolution...")
        return








def find_exploded_rois(data: Dict[str, Any]) -> None:
    """Find ROIs with exploded dF/F values"""
    
    print("=== FINDING EXPLODED dF/F ROIs ===")
    
    dff_clean = data['dFF_clean']
    n_rois = dff_clean.shape[0]
    
    # Check each ROI for extreme values
    exploded_rois = []
    extreme_rois = []
    
    for roi_idx in range(n_rois):
        roi_trace = dff_clean[roi_idx, :]
        
        roi_min = np.min(roi_trace)
        roi_max = np.max(roi_trace)
        roi_std = np.std(roi_trace)
        roi_range = roi_max - roi_min
        
        # Flag ROIs with extreme values
        if roi_max > 50 or roi_min < -50:
            exploded_rois.append({
                'roi': roi_idx,
                'min': roi_min,
                'max': roi_max,
                'std': roi_std,
                'range': roi_range
            })
        elif roi_max > 7 or roi_min < -3 or roi_std > 5:
            extreme_rois.append({
                'roi': roi_idx,
                'min': roi_min,
                'max': roi_max,
                'std': roi_std,
                'range': roi_range
            })
    
    print(f"Total ROIs: {n_rois}")
    print(f"Exploded ROIs (>50 or <-50): {len(exploded_rois)}")
    print(f"Extreme ROIs (>7, <-3, std>5): {len(extreme_rois)}")
    
    if len(exploded_rois) > 0:
        print(f"\nWorst exploded ROIs:")
        exploded_sorted = sorted(exploded_rois, key=lambda x: x['range'], reverse=True)
        for roi_info in exploded_sorted[:]:
            print(f"  ROI {roi_info['roi']:3d}: range {roi_info['range']:8.1f} "
                  f"({roi_info['min']:6.1f} to {roi_info['max']:6.1f}), std {roi_info['std']:6.1f}")
    
    if len(extreme_rois) > 0:
        print(f"\nWorst extreme ROIs:")
        extreme_sorted = sorted(extreme_rois, key=lambda x: x['range'], reverse=True)
        for roi_info in extreme_sorted[:]:
            print(f"  ROI {roi_info['roi']:3d}: range {roi_info['range']:8.1f} "
                  f"({roi_info['min']:6.1f} to {roi_info['max']:6.1f}), std {roi_info['std']:6.1f}")
    
    return exploded_rois, extreme_rois










def compute_baseline_and_dff(data: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """STEP 4B: Baseline estimation, dF/F calculation, and spike deconvolution"""
    F = data['F']
    if F is None:
        raise ValueError("F required for baseline computation - run neuropil regression first")
    Fc = data["Fc"]
    if Fc is None:
        raise ValueError("Fc required for baseline computation - run neuropil regression first")
    
    fs = float(cfg["acq"]["fs"])
    base_cfg = cfg["baseline"]
    method = base_cfg["method"]
    
    print(f"Baseline method: {method}")
    
    # Compute baseline using selected method
    if method == "custom":
        F0 = compute_percentile_baseline(F, cfg)
        Fc0 = compute_percentile_baseline(Fc, cfg)
    elif method == "suite2p":
        F0 = compute_suite2p_baseline(data, cfg)
    else:
        raise ValueError(f"Unknown baseline method: {method}. Use 'custom' or 'suite2p'")
    
    # Compute dF/F without neuropil correction
    print("  Computing dF/F...")
    eps = base_cfg.get("f0_epsilon", 1.0e-6)
    
    dF = F - F0
    # protect from divide by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        dFF = dF / (F0 + eps)
    
    data["F0"] = F0
    data["dF"] = dF
    data["dFF"] = dFF
    
    # Compute dF/F with neuropil correction
    print("  Computing dF/F...")
    eps = base_cfg.get("f0_epsilon", 1.0e-6)
    
    dFc = Fc - Fc0
    # protect from divide by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        dFFc = dFc / (Fc0 + eps)    
    
    data["Fc0"] = Fc0
    data["dFc"] = dFc
    data["dFFc"] = dFFc    
    
    # Outlier cleaning
    o_cfg = cfg["outliers"]
    if o_cfg["enable"]:
        print("  Applying Ca²⁺-aware outlier cleaning...")
        thr = o_cfg["z_thr"]
        preserve_transients = o_cfg.get("preserve_transients", True)
        
        print(f"    Z-score threshold: {thr}")
        print(f"    Preserve Ca²⁺ transients: {preserve_transients}")
        
        if preserve_transients:
            print("    Using Ca²⁺-aware outlier detection (preserves positive transients)")
        else:
            print("    Using standard outlier detection")
        
        # without neuropil correction
        dFF_clean = np.empty_like(dFF)
        total_outliers = 0
        
        for i in range(dFF.shape[0]):
            if preserve_transients:
                cleaned = fill_outliers_neighbor_mean(dFF[i], thr=thr)
            else:
                # Original method for comparison
                cleaned = fill_outliers_neighbor_mean_original(dFF[i], thr=thr)
            
            outliers = np.sum(cleaned != dFF[i])
            total_outliers += outliers
            dFF_clean[i] = cleaned
            
            # Debug first few ROIs
            if i < 3:
                print(f"      ROI {i}: {outliers} outliers cleaned")
        
        data["dFF_clean"] = dFF_clean
        print(f"    Total outliers cleaned: {total_outliers} timepoints across all ROIs (without neuropil correction)")        

        # with neuropil correction
        dFFc_clean = np.empty_like(dFFc)
        total_outliers = 0
        
        for i in range(dFFc.shape[0]):
            if preserve_transients:
                cleaned = fill_outliers_neighbor_mean(dFFc[i], thr=thr)
            else:
                # Original method for comparison
                cleaned = fill_outliers_neighbor_mean_original(dFFc[i], thr=thr)
            
            outliers = np.sum(cleaned != dFFc[i])
            total_outliers += outliers
            dFFc_clean[i] = cleaned
            
            # Debug first few ROIs
            if i < 3:
                print(f"      ROI {i}: {outliers} outliers cleaned")
        
        data["dFFc_clean"] = dFFc_clean
        print(f"    Total outliers cleaned: {total_outliers} timepoints across all ROIs (neuropil corrected)")        
    else:
        print("  Outlier cleaning disabled")
        data["dFF_clean"] = dFF.copy()
        data["dFFc_clean"] = dFFc.copy()
    
    data['dFF_smoothed'] = savgol_filter(dFF_clean, window_length=7, polyorder=3)
    data['dFFc_smoothed'] = savgol_filter(dFFc_clean, window_length=7, polyorder=3)

    # Spike deconvolution - run both versions for comparison
    deconv_cfg = cfg.get("deconvolution", {})
    if deconv_cfg.get("enable", True):
        print("  Running OASIS spike deconvolution (both versions)...")
        
        # Version 1: Suite2p batch (spikes only)
        print("    Version 1: Suite2p batch API...")
        deconv_oasis_from_Fc(data, cfg)
        
    
    # Summary statistics
    dff_final = data["dFF_clean"]
    max_dff = float(np.nanmax(np.abs(dff_final)))
    mean_f0 = float(np.nanmean(F0))
    
    print(f"Baseline, dF/F & deconvolution complete:")
    print(f"  Mean F0: {mean_f0:.1f}")
    print(f"  Max |dF/F|: {max_dff:.3f}")
    print(f"  dF/F shape: {dff_final.shape}")
    print(f"  Epsilon protection: {eps:.2e}")
    
    
    # Report on available spike data
    spike_versions = []
    if "spks_oasis" in data:
        spike_versions.append("batch")
    if "spks_oasis_recon" in data:
        spike_versions.append("reconstruction")
    
    if spike_versions:
        print(f"  Available spike data: {', '.join(spike_versions)}")    
    
    if "spks_oasis" in data:
        max_spike = float(np.nanmax(data["spks_oasis"]))
        total_spikes = int(np.sum(data["spks_oasis"] > 0))
        print(f"  Max spike rate: {max_spike:.3f}")
        print(f"  Total detected spikes: {total_spikes}")
        
    # Summary for the reconstruction version if available
    if "spks_oasis_recon" in data:
        max_spike = float(np.nanmax(data["spks_oasis_recon"]))
        total_spikes = int(np.sum(data["spks_oasis_recon"] > 0))
        print(f"  Reconstruction - Max spike rate: {max_spike:.3f}")
        print(f"  Reconstruction - Total detected spikes: {total_spikes}")
    
    return data




















# def compute_baseline_and_dff(data: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
#     """STEP 4B: Baseline estimation, dF/F calculation, and spike deconvolution"""
#     Fc = data["Fc"]
#     if Fc is None:
#         raise ValueError("Fc required for baseline computation - run neuropil regression first")
    
#     fs = float(cfg["acq"]["fs"])
#     base_cfg = cfg["baseline"]
#     method = base_cfg["method"]
    
#     print(f"Baseline method: {method}")
    
#     # Compute baseline using selected method
#     if method == "custom":
#         F0 = compute_percentile_baseline(data, cfg)
#     elif method == "suite2p":
#         F0 = compute_suite2p_baseline(data, cfg)
#     else:
#         raise ValueError(f"Unknown baseline method: {method}. Use 'custom' or 'suite2p'")
    
#     # Compute dF/F
#     print("  Computing dF/F...")
#     eps = base_cfg.get("f0_epsilon", 1.0e-6)
    
#     dF = Fc - F0
#     # protect from divide by zero
#     with np.errstate(divide='ignore', invalid='ignore'):
#         dFF = dF / (F0 + eps)
    
#     data["F0"] = F0
#     data["dF"] = dF
#     data["dFF"] = dFF
    
#     # Outlier cleaning
#     o_cfg = cfg["outliers"]
#     if o_cfg["enable"]:
#         print("  Applying Ca²⁺-aware outlier cleaning...")
#         thr = o_cfg["z_thr"]
#         preserve_transients = o_cfg.get("preserve_transients", True)
        
#         print(f"    Z-score threshold: {thr}")
#         print(f"    Preserve Ca²⁺ transients: {preserve_transients}")
        
#         if preserve_transients:
#             print("    Using Ca²⁺-aware outlier detection (preserves positive transients)")
#         else:
#             print("    Using standard outlier detection")
        
#         dFF_clean = np.empty_like(dFF)
#         total_outliers = 0
        
#         for i in range(dFF.shape[0]):
#             if preserve_transients:
#                 cleaned = fill_outliers_neighbor_mean(dFF[i], thr=thr)
#             else:
#                 # Original method for comparison
#                 cleaned = fill_outliers_neighbor_mean_original(dFF[i], thr=thr)
            
#             outliers = np.sum(cleaned != dFF[i])
#             total_outliers += outliers
#             dFF_clean[i] = cleaned
            
#             # Debug first few ROIs
#             if i < 3:
#                 print(f"      ROI {i}: {outliers} outliers cleaned")
        
#         data["dFF_clean"] = dFF_clean
#         print(f"    Total outliers cleaned: {total_outliers} timepoints across all ROIs")
#     else:
#         print("  Outlier cleaning disabled")
#         data["dFF_clean"] = dFF.copy()
    
#     # Spike deconvolution - run both versions for comparison
#     deconv_cfg = cfg.get("deconvolution", {})
#     if deconv_cfg.get("enable", True):
#         print("  Running OASIS spike deconvolution (both versions)...")
        
#         # Version 1: Suite2p batch (spikes only)
#         print("    Version 1: Suite2p batch API...")
#         deconv_oasis_from_Fc(data, cfg)
        
        
#         # # Version 2: Individual trace reconstruction (full components)
#         # print("    Version 2: Individual trace reconstruction...")
#         # deconv_oasis_from_Fc_recon(data, cfg)
        
#     #     # Compare results if both succeeded
#     #     if "spks_oasis" in data and "spks_oasis_recon" in data:
#     #         print("  Comparing deconvolution methods...")
#     #         spks_batch = data["spks_oasis"]
#     #         spks_recon = data["spks_oasis_recon"]
            
#     #         # Basic comparison metrics
#     #         corr_coef = np.corrcoef(spks_batch.flatten(), spks_recon.flatten())[0,1]
#     #         batch_total = np.sum(spks_batch > 0)
#     #         recon_total = np.sum(spks_recon > 0)
            
#     #         print(f"    Spike correlation: {corr_coef:.4f}")
#     #         print(f"    Batch spikes: {batch_total}, Recon spikes: {recon_total}")
#     #         print(f"    Spike count ratio: {recon_total/max(batch_total,1):.3f}")
            
#     #         # Check if we got the extra components from reconstruction
#     #         if "C_oasis_recon" in data:
#     #             print(f"    Reconstruction provides: spikes, denoised calcium, baselines, decay coefficients")
#     #         else:
#     #             print(f"    Reconstruction failed to provide additional components")
#     # else:
#     #     print("  Spike deconvolution disabled")
    
#     # Summary statistics
#     dff_final = data["dFF_clean"]
#     max_dff = float(np.nanmax(np.abs(dff_final)))
#     mean_f0 = float(np.nanmean(F0))
    
#     print(f"Baseline, dF/F & deconvolution complete:")
#     print(f"  Mean F0: {mean_f0:.1f}")
#     print(f"  Max |dF/F|: {max_dff:.3f}")
#     print(f"  dF/F shape: {dff_final.shape}")
#     print(f"  Epsilon protection: {eps:.2e}")
    
    
#     # Report on available spike data
#     spike_versions = []
#     if "spks_oasis" in data:
#         spike_versions.append("batch")
#     if "spks_oasis_recon" in data:
#         spike_versions.append("reconstruction")
    
#     if spike_versions:
#         print(f"  Available spike data: {', '.join(spike_versions)}")    
    
#     if "spks_oasis" in data:
#         max_spike = float(np.nanmax(data["spks_oasis"]))
#         total_spikes = int(np.sum(data["spks_oasis"] > 0))
#         print(f"  Max spike rate: {max_spike:.3f}")
#         print(f"  Total detected spikes: {total_spikes}")
        
#     # Summary for the reconstruction version if available
#     if "spks_oasis_recon" in data:
#         max_spike = float(np.nanmax(data["spks_oasis_recon"]))
#         total_spikes = int(np.sum(data["spks_oasis_recon"] > 0))
#         print(f"  Reconstruction - Max spike rate: {max_spike:.3f}")
#         print(f"  Reconstruction - Total detected spikes: {total_spikes}")
    
#     return data

def deconv_oasis_from_Fc(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """
    OASIS deconvolution using Suite2p's batch API.
    Uses neuropil-corrected fluorescence with optional F0 detrending.
    Stores: C_oasis (denoised), spks_oasis (spikes), oasis_b, oasis_g, oasis_lam
    """
    print("  OASIS deconvolution setup...")
    
    # Get configuration parameters
    deconv_cfg = cfg.get("deconvolution", {})
    tau_s = deconv_cfg.get("tau_s", 0.7)  # Default 0.7s decay time
    batch_size = deconv_cfg.get("batch_size")  # None = auto
    detrend_with_f0 = deconv_cfg.get("detrend_with_f0", True)
    
    fs = float(cfg["acq"]["fs"])
    Fc = data["Fc"]
    
    if Fc is None:
        raise ValueError("Fc required for deconvolution - run neuropil regression first")
    
    print(f"  Deconvolution parameters:")
    print(f"    Sampling rate: {fs:.1f} Hz")
    print(f"    Tau (decay time): {tau_s}s")
    print(f"    Detrend with F0: {detrend_with_f0}")
    
    # Prepare fluorescence data
    F = Fc.astype(np.float32)
    N, T = F.shape
    
    # Optional gentle detrending with F0
    if detrend_with_f0 and "F0" in data:
        print("  Applying gentle F0 detrending...")
        F0 = data["F0"]
        # Remove slow trend but keep additive scale (not dF/F)
        median_f0 = np.median(F0, axis=1, keepdims=True)
        F = F - F0 + median_f0
        print(f"    Detrended signal range: [{F.min():.1f}, {F.max():.1f}]")
    else:
        print("  Using raw Fc (no F0 detrending)")
    
    # Set batch size
    if batch_size is None:
        batch_size = min(T, 20000)  # Safe default for memory
        print(f"    Auto batch size: {batch_size} frames")
    else:
        print(f"    Batch size: {batch_size} frames")
    
    print(f"  Processing {N} ROIs x {T} timepoints...")
    
    try:
        # Suite2p OASIS batch deconvolution
        S = oasis(F=F, batch_size=batch_size, tau=tau_s, fs=fs)
        
        # Store results
        data["spks_oasis"] = S.astype(np.float32)   # Inferred spikes
        
        # Compute statistics
        total_spikes = np.sum(S > 0)
        mean_spike_rate = total_spikes / (N * T / fs)  # spikes per ROI per second
        max_spike = np.max(S)
        
        print(f"  Deconvolution results:")
        print(f"    Total spikes detected: {total_spikes}")
        print(f"    Mean spike rate: {mean_spike_rate:.3f} spikes/ROI/s")
        print(f"    Max spike amplitude: {max_spike:.3f}")
        print("  OASIS deconvolution complete")
        
    except Exception as e:
        print(f"  ERROR: OASIS deconvolution failed: {e}")
        print("  Skipping spike deconvolution...")
        # Don't store partial results on failure
        return

# ==================== HELPER FUNCTIONS ====================
def fill_outliers_neighbor_mean_original(x: np.ndarray, thr: float) -> np.ndarray:
    """Fill outliers using neighbor mean"""
    y = x.copy()
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med)) + 1e-9
    z = 0.6745 * (y - med) / mad
    mask = np.abs(z) > thr
    
    if not mask.any():
        return y
    
    idx = np.where(mask)[0]
    cuts = np.where(np.diff(idx) > 1)[0] + 1
    segments = np.split(idx, cuts)
    T = y.size
    
    for seg in segments:
        if seg.size == 0:
            continue
        s, e = seg[0], seg[-1]
        left_val = y[s-1] if s-1 >= 0 and not mask[s-1] else None
        right_val = y[e+1] if e+1 < T and not mask[e+1] else None
        
        if left_val is not None and right_val is not None:
            repl = 0.5 * (left_val + right_val)
        elif left_val is not None:
            repl = left_val
        elif right_val is not None:
            repl = right_val
        else:
            repl = 0.0
        y[s:e+1] = repl
    
    return y


def fill_outliers_neighbor_mean(x: np.ndarray, thr: float) -> np.ndarray:
    """Fill outliers using neighbor mean - UPDATED to preserve Ca²⁺ transients"""
    y = x.copy()
    
    # Use more robust baseline estimation to avoid flagging transients as outliers
    # Use lower percentiles for baseline (exclude transients from noise estimation)
    baseline_est = np.percentile(y[np.isfinite(y)], [5, 10, 20, 30])
    
    # Use the 10th percentile as baseline (excludes most transients)
    baseline = baseline_est[1]  # 10th percentile
    
    # Compute MAD relative to baseline (not median)
    residuals = y - baseline
    mad = np.nanmedian(np.abs(residuals - np.median(residuals))) + 1e-9
    
    # Only flag as outliers if they're NEGATIVE deviations beyond threshold
    # OR extremely large positive deviations (likely artifacts, not Ca transients)
    z = 0.6745 * residuals / mad
    
    # Modified outlier detection:
    # 1. Negative deviations > threshold (baseline artifacts)
    # 2. Extremely large positive deviations (> 2x threshold) - likely imaging artifacts
    negative_outliers = (z < -thr)  # Negative outliers
    extreme_positive = (z > 2 * thr)  # Only extreme positive outliers
    
    mask = negative_outliers | extreme_positive
    
    print(f"        Outlier stats: {negative_outliers.sum()} negative, {extreme_positive.sum()} extreme positive, {mask.sum()} total")
    
    if not mask.any():
        return y
    
    # Rest of the function remains the same
    idx = np.where(mask)[0]
    cuts = np.where(np.diff(idx) > 1)[0] + 1
    segments = np.split(idx, cuts)
    T = y.size
    
    for seg in segments:
        if seg.size == 0:
            continue
        s, e = seg[0], seg[-1]
        left_val = y[s-1] if s-1 >= 0 and not mask[s-1] else None
        right_val = y[e+1] if e+1 < T and not mask[e+1] else None
        
        if left_val is not None and right_val is not None:
            repl = 0.5 * (left_val + right_val)
        elif left_val is not None:
            repl = left_val
        elif right_val is not None:
            repl = right_val
        else:
            repl = baseline  # Use baseline instead of 0
        y[s:e+1] = repl
    
    return y




def print_label_summary(data: Dict[str, Any]):
    print("\n=== LABEL SUMMARY ===")
    labs = data.get('roi_labels')
    if not labs:
        print("No labels found")
        return
    
    from collections import Counter
    c = Counter(labs)
    total = len(labs)
    
    print("ROI classification results:")
    for label, count in c.items():
        pct = 100 * count / total
        print(f"  {label}: {count}/{total} ({pct:.1f}%)")

# ==================== VISUALIZATION ====================
def _build_roi_outline(data: Dict[str, Any], idx: int, shape: Tuple[int,int]) -> np.ndarray:
    H, W = shape
    st = data['stat'][idx]
    ypix, xpix, lam = _roi_pixels_from_stat(st)
    ypix, xpix, lam = _clip_roi_pixels_to_image(ypix, xpix, lam, H, W)
    
    if ypix.size == 0:
        return np.zeros((H, W), dtype=bool)
    
    m = np.zeros((H, W), dtype=bool)
    m[ypix, xpix] = True
    
    from scipy.ndimage import binary_erosion
    inner = binary_erosion(m, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]), border_value=0)
    outline = m & (~inner)
    return outline

def generate_refined_label_overlay(data: Dict[str, Any], cfg: Dict[str, Any], roi_indices=None, show: bool = True, save: bool = False):
    """Generate refined overlay with uncertain class in yellow"""
    print("\n=== GENERATING REFINED LABEL OVERLAY ===")
    
    A = data.get('anatomy_image')
    labels = data.get('roi_labels')
    
    if A is None:
        raise ValueError("anatomy_image missing")
    if labels is None:
        raise ValueError("roi_labels missing")
    
    ov_cfg = cfg['overlay']
    H, W = A.shape
    
    if roi_indices is None:
        roi_indices = range(len(data['stat']))
        print(f"Overlaying all {len(roi_indices)} ROIs")
    else:
        print(f"Overlaying {len(roi_indices)} selected ROIs")
    
    # Normalize background
    lo, hi = np.percentile(A, (ov_cfg['bg_pmin'], ov_cfg['bg_pmax']))
    bg = np.clip((A - lo) / (hi - lo + 1e-9), 0, 1)
    canvas = np.dstack([bg, bg, bg])
    
    print(f"Background normalization: [{lo:.1f}, {hi:.1f}] to [0, 1]")
    
    # Extended color mapping including uncertain
    color_map = {
        'soma': np.array(ov_cfg['color_soma'], dtype=float) / 255.0,
        'soma_cp': np.array(ov_cfg.get('color_soma_cp', ov_cfg['color_soma']), dtype=float) / 255.0,
        'process': np.array(ov_cfg['color_process'], dtype=float) / 255.0,
        'uncertain': np.array([255, 255, 0], dtype=float) / 255.0  # Yellow for uncertain
    }
    
    alpha = float(ov_cfg['alpha'])
    lw = int(ov_cfg['line_width'])
    
    print(f"Overlay parameters: alpha={alpha}, line_width={lw}")
    print(f"Color mapping: soma=red, process=cyan, uncertain=yellow")
    
    # Draw outlines
    label_counts = {}
    for rid in roi_indices:
        if rid >= len(data['stat']) or rid >= len(labels):
            continue
        
        label = labels[rid]
        label_counts[label] = label_counts.get(label, 0) + 1
        
        color = color_map.get(label, color_map['uncertain'])
        outline = _build_roi_outline(data, rid, (H, W))
        
        if not outline.any():
            continue
        
        # Optional line thickening
        if lw > 1:
            for _ in range(lw-1):
                outline = binary_dilation(outline)
        
        # Blend color
        for c in range(3):
            canvas[..., c][outline] = (1 - alpha) * canvas[..., c][outline] + alpha * color[c]
    
    print(f"Drew outlines for: {label_counts}")
    
    # Create figure
    fig_w, fig_h = ov_cfg['fig_size']
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(canvas, interpolation='nearest')
    ax.set_title("Refined ROI Classification Overlay\n(Red=Soma, Cyan=Process, Yellow=Uncertain)")
    ax.axis('off')
    
    if save:
        out_dir = ov_cfg['save_dir']
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, 'refined_' + ov_cfg['overlay_filename'])
        fig.savefig(save_path, dpi=ov_cfg['dpi'])
        print(f"Refined overlay saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    print("Refined label overlay complete")
    return fig


def generate_label_overlay(data: Dict[str, Any], cfg: Dict[str, Any], roi_indices=None, show: bool = True, save: bool = False):
    """Generate composite ROI overlay"""
    print("\n=== GENERATING LABEL OVERLAY ===")
    
    A = data.get('anatomy_image')
    labels = data.get('roi_labels')
    
    if A is None:
        raise ValueError("anatomy_image missing")
    if labels is None:
        raise ValueError("roi_labels missing")
    
    ov_cfg = cfg['overlay']
    H, W = A.shape
    
    if roi_indices is None:
        roi_indices = range(len(data['stat']))
        print(f"Overlaying all {len(roi_indices)} ROIs")
    else:
        print(f"Overlaying {len(roi_indices)} selected ROIs")
    
    # Normalize background
    lo, hi = np.percentile(A, (ov_cfg['bg_pmin'], ov_cfg['bg_pmax']))
    bg = np.clip((A - lo) / (hi - lo + 1e-9), 0, 1)
    canvas = np.dstack([bg, bg, bg])
    
    print(f"Background normalization: [{lo:.1f}, {hi:.1f}] to [0, 1]")
    
    # Color mapping
    color_map = {
        'soma': np.array(ov_cfg['color_soma'], dtype=float) / 255.0,
        'soma_cp': np.array(ov_cfg.get('color_soma_cp', ov_cfg['color_soma']), dtype=float) / 255.0,
        'process': np.array(ov_cfg['color_process'], dtype=float) / 255.0
    }
    
    alpha = float(ov_cfg['alpha'])
    lw = int(ov_cfg['line_width'])
    
    print(f"Overlay parameters: alpha={alpha}, line_width={lw}")
    
    # Draw outlines
    label_counts = {}
    for rid in roi_indices:
        if rid >= len(data['stat']) or rid >= len(labels):
            continue
        
        label = labels[rid]
        label_counts[label] = label_counts.get(label, 0) + 1
        
        color = color_map.get(label, color_map['process'])
        outline = _build_roi_outline(data, rid, (H, W))
        
        if not outline.any():
            continue
        
        # Optional line thickening
        if lw > 1:
            from scipy.ndimage import binary_dilation
            for _ in range(lw-1):
                outline = binary_dilation(outline)
        
        # Blend color
        for c in range(3):
            canvas[..., c][outline] = (1 - alpha) * canvas[..., c][outline] + alpha * color[c]
    
    print(f"Drew outlines for: {label_counts}")
    
    # Create figure
    fig_w, fig_h = ov_cfg['fig_size']
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    # ax.imshow(canvas, interpolation='nearest')
    ax.set_title("ROI Classification Overlay")
    ax.axis('off')
    
    if save:
        out_dir = ov_cfg['save_dir']
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, ov_cfg['overlay_filename'])
        fig.savefig(save_path, dpi=ov_cfg['dpi'])
        print(f"Overlay saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    print("Label overlay complete")
    return fig

# def prepare_review_cache(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
#     """Prepare downsampled data for review plots"""
#     print("\n=== PREPARING REVIEW CACHE ===")
    
#     rv = cfg['review']
#     max_pts = int(rv['max_points_full'])
    
#     required = ['F', 'Fc', 'F0', 'dFF']
#     for key in required:
#         if key not in data:
#             raise ValueError(f"Missing {key} - run processing steps first")
    
#     fs = float(cfg['acq']['fs'])
#     T = data['F'].shape[1]
#     decim = max(1, int(np.ceil(T / max_pts)))
#     sl = slice(0, T, decim)
    
#     print(f"Decimation: {T} to {len(range(0, T, decim))} points (factor: {decim})")
    
#     review = {
#         'fs': fs, 'decim': decim, 'T': T,
#         't_full': np.arange(T) / fs,
#         't_ds': np.arange(0, T, decim) / fs,
#         'F_ds': data['F'][:, sl],
#         'Fneu_ds': data.get('Fneu')[:, sl] if data.get('Fneu') is not None else None,
#         'Fc_ds': data['Fc'][:, sl],
#         'F0_ds': data['F0'][:, sl],
#         'dFF_ds': data['dFF'][:, sl],
#     }
    
#     data['review'] = review
#     print(f"Review cache prepared: {review['F_ds'].shape}")

def prepare_review_cache(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """Prepare downsampled data for review plots"""
    print("\n=== PREPARING REVIEW CACHE ===")
    
    rv = cfg['review']
    max_pts = int(rv['max_points_full'])
    
    required = ['F', 'Fc', 'F0', 'dFF']
    for key in required:
        if key not in data:
            raise ValueError(f"Missing {key} - run processing steps first")
    
    fs = float(cfg['acq']['fs'])
    T = data['F'].shape[1]
    decim = max(1, int(np.ceil(T / max_pts)))
    sl = slice(0, T, decim)
    
    print(f"Decimation: {T} to {len(range(0, T, decim))} points (factor: {decim})")
    
    # Check for spike data availability
    spike_version = None
    if 'spks_oasis' in data:
        spike_version = 'spks_oasis'
    elif 'spks_oasis_recon' in data:
        spike_version = 'spks_oasis_recon'
    
    if spike_version:
        print(f"Including spike data: {spike_version}")
    else:
        print("No spike data available for review cache")
    
    review = {
        'fs': fs, 'decim': decim, 'T': T,
        't_full': np.arange(T) / fs,
        't_ds': np.arange(0, T, decim) / fs,
        'F_ds': data['F'][:, sl],
        'Fneu_ds': data.get('Fneu')[:, sl] if data.get('Fneu') is not None else None,
        'Fc_ds': data['Fc'][:, sl],
        'F0_ds': data['F0'][:, sl],
        'dFF_ds': data['dFF'][:, sl],
        'dFF_smoothed_ds': data['dFF_smoothed'][:, sl],
        'spks_ds': data[spike_version][:, sl] if spike_version else None,
        'spike_version': spike_version,
    }
    
    data['review'] = review
    print(f"Review cache prepared: {review['F_ds'].shape}")
    if spike_version:
        print(f"  Spike data cached: {review['spks_ds'].shape} ({spike_version})")


# ==================== SUMMARY FUNCTIONS ====================
# def summarize(data: Dict[str, Any], cfg: Dict[str, Any], n: int = 5):
#     print("\n=== PROCESSING SUMMARY ===")
    
#     dff = data.get("dFF")
#     if dff is None:
#         print("No dFF computed yet")
#         return
    
#     N, T = dff.shape
#     fs = cfg['acq']['fs']
#     duration = T / fs
    
#     print(f"Dataset: {N} ROIs x {T} timepoints ({duration:.1f}s @ {fs:.1f} Hz)")
    
#     if 'roi_labels' in data:
#         from collections import Counter
#         counts = Counter(data['roi_labels'])
#         print(f"Classifications: {dict(counts)}")
    
#     if 'neuropil_a' in data:
#         a = data['neuropil_a']
#         print(f"Neuropil alpha: {a.mean():.3f} +/- {a.std():.3f} (range: [{a.min():.3f}, {a.max():.3f}])")
    
#     if 'dFF_clean' in data:
#         dff_clean = data['dFF_clean']
#         max_dff = np.nanmax(np.abs(dff_clean))
#         mean_dff = np.nanmean(np.abs(dff_clean))
#         print(f"dF/F: max |dF/F| = {max_dff:.3f}, mean |dF/F| = {mean_dff:.3f}")
        
#         if n > 0:
#             print(f"Sample |dF/F| (first {n}): {np.round(np.nanmax(np.abs(dff_clean[:n]), axis=1), 3)}")


def summarize(data: Dict[str, Any], cfg: Dict[str, Any], n: int = 5):
    print("\n=== PROCESSING SUMMARY ===")
    
    dff = data.get("dFF")
    if dff is None:
        print("No dFF computed yet")
        return
    
    N, T = dff.shape
    fs = cfg['acq']['fs']
    duration = T / fs
    
    print(f"Dataset: {N} ROIs x {T} timepoints ({duration:.1f}s @ {fs:.1f} Hz)")
    
    if 'roi_labels' in data:
        from collections import Counter
        counts = Counter(data['roi_labels'])
        print(f"Classifications: {dict(counts)}")
    
    if 'neuropil_a' in data:
        a = data['neuropil_a']
        print(f"Neuropil alpha: {a.mean():.3f} +/- {a.std():.3f} (range: [{a.min():.3f}, {a.max():.3f}])")
    
    if 'dFF_clean' in data:
        dff_clean = data['dFF_clean']
        max_dff = np.nanmax(np.abs(dff_clean))
        mean_dff = np.nanmean(np.abs(dff_clean))
        print(f"dF/F: max |dF/F| = {max_dff:.3f}, mean |dF/F| = {mean_dff:.3f}")
        
        if n > 0:
            print(f"Sample |dF/F| (first {n}): {np.round(np.nanmax(np.abs(dff_clean[:n]), axis=1), 3)}")
    
    # Add Ca transient summary
    if 'qc_metrics' in data and 'ca_event_counts' in data['qc_metrics']:
        qc = data['qc_metrics']
        total_events = qc['ca_event_counts'].sum()
        mean_rate = qc['ca_event_rates_hz'].mean()
        active_rois = np.sum(qc['ca_event_counts'] > 0)
        mean_activity = qc['ca_activity_fraction'].mean()
        
        print(f"Ca transients: {total_events} events, {mean_rate:.2f} events/s mean rate")
        print(f"  Active ROIs: {active_rois}/{N} ({100*active_rois/N:.1f}%)")
        print(f"  Mean activity fraction: {100*mean_activity:.1f}%")
        
        if total_events > 0:
            mean_amp = qc['ca_mean_amplitudes'][qc['ca_event_counts'] > 0].mean()
            print(f"  Mean amplitude: {mean_amp:.3f} ΔF/F")
    
    # Add spike summary if available
    if 'qc_metrics' in data and 'spike_rate_mean_hz' in data['qc_metrics']:
        qc = data['qc_metrics']
        print(f"Spike detection: mean rate {qc['spike_rate_mean_hz'].mean():.2f} Hz")
        print(f"  Total spike onsets: {qc['spike_count_onsets'].sum()}")


def quick_plot(data: Dict[str, Any], cfg: Dict[str, Any], roi: int = 0):
    print(f"\n=== QUICK PLOT ROI {roi} ===")
    
    if "dFF" not in data:
        print("No dFF data available")
        return
    
    if roi >= data["dFF"].shape[0]:
        print(f"ROI {roi} out of range (max: {data['dFF'].shape[0]-1})")
        return
    
    fig_w = cfg.get('plot', {}).get('width', 8)
    fig_h = cfg.get('plot', {}).get('height', 3)
    
    plt.figure(figsize=(fig_w, fig_h))
    plt.plot(data["dFF"][roi], lw=0.6, label="dFF")
    
    if "dFF_clean" in data:
        plt.plot(data["dFF_clean"][roi], lw=0.6, label="clean", alpha=0.7)
    
    plt.title(f"ROI {roi}")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    
    print(f"Quick plot displayed")

# ==================== OTHER VISUALIZATION FUNCTIONS ====================
# def _segments_to_idx(segments: list, fs: float, T: int):
#     out = []
#     for (a, b) in segments:
#         a = float(a); b = float(b)
#         if b <= a: continue
#         i0 = int(round(a * fs))
#         i1 = int(round(b * fs))
#         if i0 >= T: continue
#         i1 = min(i1, T)
#         if i1 - i0 < 2: continue
#         t_local = (np.arange(i0, i1) - i0) / fs
#         out.append((slice(i0, i1), t_local, (a, b)))
#     return out

def _segments_to_idx(segments: list, fs: float, T: int):
    out = []
    for (a, b) in segments:
        a = float(a); b = float(b)
        if b <= a: continue
        i0 = int(round(a * fs))
        i1 = int(round(b * fs))
        if i0 >= T: continue
        i1 = min(i1, T)
        if i1 - i0 < 2: continue
        
        # FIX: Use absolute time, not relative to segment start
        t_local = np.arange(i0, i1) / fs  # This gives absolute time (300-340s)
        # OLD: t_local = (np.arange(i0, i1) - i0) / fs  # This gave relative time (0-40s)
        
        out.append((slice(i0, i1), t_local, (a, b)))
    return out






# def detect_ca_transients(x, fs,
#                          smooth_sigma_s=0.10,      # light smoothing (s)
#                          k_on=2.0, k_off=0.8,      # LOWERED thresholds 
#                          rise_max_s=0.50,          # INCREASED rise time allowance
#                          peak_win_s=1.50,          # INCREASED peak search window
#                          decay_max_s=4.0,          # INCREASED decay allowance
#                          min_amp=0.15,             # LOWERED amplitude threshold
#                          use_z=False,              # set True to work purely in z-units
#                          # REFINED: Biological constraints
#                          max_decay_slope_per_s=4.0,    # RELAXED - allow faster decays
#                          min_series_peak_dff=0.25,     # LOWERED - less strict peak requirement
#                          series_timeout_s=5.0,         # INCREASED - more time to find peaks
#                          merge_gap_s=0.3,              # SHORTENED - less aggressive merging
#                          min_snr=1.2,                  # LOWERED - more permissive SNR
#                          # NEW: Shape-based filtering
#                          min_event_duration_s=0.05,    # Minimum event duration
#                          max_baseline_drift_per_s=0.5, # Max allowed baseline drift during event
#                          require_clear_peak=True):     # Require obvious peak structure
#     """
#     Enhanced Ca²⁺ transient detection with better sensitivity/specificity balance
    
#     Returns:
#       events: list of dicts with {t_on, t_peak, t_off, amp, rise_s, decay_s, area}
#       mask:   boolean array True during accepted events
#     """
#     x = np.asarray(x, float)
#     T = x.size

#     if T < 2:
#         return [], np.zeros(T, dtype=bool)
    
#     # Check for invalid traces
#     if np.all(np.isnan(x)) or np.all(x == x[0]):
#         return [], np.zeros(T, dtype=bool)

#     # Light smoothing
#     if smooth_sigma_s and smooth_sigma_s > 0:
#         sigma_frames = max(1, int(round(smooth_sigma_s * fs)))
#         x_s = gaussian_filter1d(x, sigma=sigma_frames)
#     else:
#         x_s = x.copy()

#     # IMPROVED: More robust noise estimation
#     try:
#         valid_mask = np.isfinite(x_s)
#         if not np.any(valid_mask):
#             return [], np.zeros(T, dtype=bool)
        
#         x_valid = x_s[valid_mask]
        
#         if len(x_valid) < 10 or np.std(x_valid) < 1e-12:
#             return [], np.zeros(T, dtype=bool)
        
#         # Use multiple percentiles for more robust baseline estimation
#         p10 = np.percentile(x_valid, 10)
#         p20 = np.percentile(x_valid, 20) 
#         p30 = np.percentile(x_valid, 30)
        
#         # Use the most stable baseline region
#         baseline_candidates = [p10, p20, p30]
#         residuals = []
#         for p in baseline_candidates:
#             r = x_valid - p
#             mad = np.median(np.abs(r - np.median(r)))
#             residuals.append(mad)
        
#         # Choose baseline with smallest MAD (most stable)
#         best_baseline = baseline_candidates[np.argmin(residuals)]
#         r = x_valid - best_baseline
#         mad = np.median(np.abs(r - np.median(r)))
        
#         if mad < 1e-12:
#             mad = np.std(x_valid) / 1.4826
#             if mad < 1e-12:
#                 return [], np.zeros(T, dtype=bool)
        
#         sigma = 1.4826 * mad
        
#     except (ValueError, RuntimeError) as e:
#         print(f"      Error in improved noise estimation: {e}")
#         return [], np.zeros(T, dtype=bool)

#     # Adaptive thresholds based on signal characteristics
#     try:
#         if use_z:
#             med_val = np.median(x_s[valid_mask])
#             z = (x_s - med_val) / (sigma + 1e-12)
#             th_on, th_off = k_on, k_off
#             y = z
#             amp_thr = 0.0
#         else:
#             # Use the robust baseline estimate
#             med_val = best_baseline
#             y = x_s
#             th_on = med_val + k_on * sigma
#             th_off = med_val + k_off * sigma
#             amp_thr = min_amp
            
#     except (ValueError, RuntimeError) as e:
#         print(f"      Error in threshold calculation: {e}")
#         return [], np.zeros(T, dtype=bool)

#     # Convert time parameters to samples
#     rise_max = int(round(rise_max_s * fs))
#     peak_win = int(round(peak_win_s * fs))
#     decay_max = int(round(decay_max_s * fs))
#     series_timeout = int(round(series_timeout_s * fs))
#     merge_gap = int(round(merge_gap_s * fs))
#     min_duration = int(round(min_event_duration_s * fs))

#     events = []
#     mask = np.zeros(T, dtype=bool)
#     i = 1
    
#     try:
#         while i < T:
#             # Skip NaN values
#             if not np.isfinite(y[i-1]) or not np.isfinite(y[i]):
#                 i += 1
#                 continue
                
#             # ONSET: upward crossing of th_on
#             if y[i-1] < th_on and y[i] >= th_on:
#                 i_on = i

#                 # Find baseline (last crossing below th_off)
#                 j = i_on
#                 while j > 0 and np.isfinite(y[j-1]) and y[j-1] > th_off:
#                     j -= 1
#                 i_base = j
                
#                 if not np.isfinite(y[i_base]):
#                     i += 1
#                     continue
                    
#                 baseline = y[i_base]

#                 # Rise time check (more lenient)
#                 j = i_on
#                 while j > 0 and np.isfinite(y[j-1]) and y[j-1] > th_off:
#                     j -= 1
#                 rise_len = i_on - j
#                 if rise_len > rise_max:
#                     i += 1
#                     continue

#                 # RELAXED: Peak search in event series
#                 i_end_search = min(T, i_on + series_timeout)
#                 max_peak_in_series = np.nanmax(y[i_on:i_end_search]) if i_end_search > i_on else y[i_on]
#                 series_amp = max_peak_in_series - baseline
                
#                 if series_amp < min_series_peak_dff:
#                     i += 1
#                     continue

#                 # Find peak within initial window
#                 i_end_peak = min(T, i_on + max(peak_win, 1))
#                 peak_segment = y[i_on:i_end_peak]
#                 valid_peak_mask = np.isfinite(peak_segment)
#                 if not np.any(valid_peak_mask):
#                     i += 1
#                     continue
                
#                 local_peak_idx = np.nanargmax(peak_segment)
#                 i_peak = i_on + local_peak_idx
#                 peak_val = y[i_peak]
                
#                 if not np.isfinite(peak_val):
#                     i += 1
#                     continue
                
#                 amp = peak_val - baseline
#                 if amp < amp_thr:
#                     i += 1
#                     continue

#                 # Find decay with merging allowed
#                 i_search_end = min(T, i_on + max(decay_max, 1))
#                 k = i_peak
#                 i_off = None
                
#                 while k < i_search_end:
#                     if not np.isfinite(y[k]):
#                         k += 1
#                         continue
                        
#                     # Allow re-crossings above onset (complex events)
#                     if y[k] >= th_on and y[k] > peak_val:
#                         i_peak, peak_val = k, y[k]
#                         amp = peak_val - baseline
                    
#                     if y[k] < th_off:
#                         i_off = k
#                         break
#                     k += 1

#                 if i_off is None:
#                     i += 1
#                     continue

#                 # NEW: Event duration check
#                 event_duration_s = (i_off - i_base) / fs
#                 if event_duration_s < min_event_duration_s:
#                     i += 1
#                     continue

#                 # RELAXED: Decay slope constraint
#                 decay_s = (i_off - i_peak) / fs
#                 if decay_s > 0:
#                     decay_slope = (peak_val - y[i_off]) / decay_s
#                     if decay_slope > max_decay_slope_per_s:
#                         i += 1
#                         continue

#                 # NEW: Baseline stability check during event
#                 if max_baseline_drift_per_s > 0:
#                     event_segment = y[i_base:i_off]
#                     if len(event_segment) > 3:
#                         # Check if baseline drifts too much during event
#                         baseline_trend = np.polyfit(np.arange(len(event_segment)), event_segment, 1)[0]
#                         baseline_drift_rate = abs(baseline_trend * fs)
#                         if baseline_drift_rate > max_baseline_drift_per_s:
#                             i += 1
#                             continue

#                 # NEW: Peak prominence check (clear peak requirement)
#                 if require_clear_peak and len(event_segment) > 5:
#                     # Require peak to be clearly above surrounding region
#                     peak_rel_idx = i_peak - i_base
#                     if peak_rel_idx > 2 and peak_rel_idx < len(event_segment) - 2:
#                         surrounding = np.concatenate([
#                             event_segment[max(0, peak_rel_idx-2):peak_rel_idx],
#                             event_segment[peak_rel_idx+1:min(len(event_segment), peak_rel_idx+3)]
#                         ])
#                         if len(surrounding) > 0:
#                             peak_prominence = peak_val - np.max(surrounding)
#                             if peak_prominence < 0.3 * amp:  # Peak must be 30% above surroundings
#                                 i += 1
#                                 continue

#                 # RELAXED: SNR requirement
#                 if min_snr > 0:
#                     pre_event_start = max(0, i_base - 30)  # Shorter pre-event window
#                     if i_base > pre_event_start:
#                         noise_segment = y[pre_event_start:i_base]
#                         noise_std = np.nanstd(noise_segment[np.isfinite(noise_segment)])
#                         if noise_std > 0:
#                             local_snr = amp / noise_std
#                             if local_snr < min_snr:
#                                 i += 1
#                                 continue

#                 # Calculate event metrics
#                 rise_s = (i_peak - i_base) / fs
#                 decay_s = (i_off - i_peak) / fs
                
#                 # Calculate area
#                 event_segment = y[i_base:i_off]
#                 valid_event_mask = np.isfinite(event_segment)
#                 if np.any(valid_event_mask):
#                     area_values = np.maximum(event_segment[valid_event_mask] - baseline, 0)
#                     area = np.trapz(area_values, dx=1/fs)
#                 else:
#                     area = 0.0

#                 # REFINED: Event merging (shorter gap = less aggressive)
#                 if events and (i_on - events[-1]['i_off']) <= merge_gap:
#                     # Merge with previous event
#                     last_event = events[-1]
                    
#                     # Update peak if current is higher
#                     if peak_val > y[last_event['i_peak']]:
#                         events[-1].update({
#                             't_peak': i_peak/fs,
#                             'i_peak': i_peak,
#                             'amp': max(amp, last_event['amp'])
#                         })
                    
#                     # Extend off time
#                     events[-1]['t_off'] = i_off/fs
#                     events[-1]['i_off'] = i_off
#                     events[-1]['decay_s'] = (i_off - events[-1]['i_peak']) / fs
                    
#                     # Recalculate area
#                     merged_segment = y[last_event['i_base']:i_off]
#                     valid_merged_mask = np.isfinite(merged_segment)
#                     if np.any(valid_merged_mask):
#                         merged_area_values = np.maximum(merged_segment[valid_merged_mask] - last_event['baseline'], 0)
#                         events[-1]['area'] = np.trapz(merged_area_values, dx=1/fs)
                    
#                     # Update mask
#                     mask[last_event['i_base']:i_off] = True
                    
#                 else:
#                     # NEW (FIXED):
#                     events.append(dict(
#                         t_on=i_base/fs, t_peak=i_peak/fs, t_off=i_off/fs,  # Use i_base for onset time
#                         amp=amp, rise_s=rise_s, decay_s=decay_s, area=area,
#                         i_on=i_on, i_peak=i_peak, i_off=i_off, baseline=baseline,
#                         i_base=i_base
#                     ))
#                     mask[i_base:i_off] = True

#                 i = i_off + 1
#             else:
#                 i += 1

#     except Exception as e:
#         print(f"      Error in event detection loop: {e}")
#         pass

#     return events, mask


# Replace the main detect_ca_transients function with the working algorithm
def detect_ca_transients(x, fs,
                         smooth_sigma_s=0.10,      # light smoothing (s)
                         k_on=2.0, k_off=0.8,      # onset/offset thresholds
                         rise_max_s=0.50,          # rise time allowance
                         peak_win_s=1.50,          # peak search window
                         decay_max_s=4.0,          # decay allowance
                         min_amp=0.15,             # minimum amplitude
                         use_z=False,              # set True to work purely in z-units
                         # Biological constraints
                         max_decay_slope_per_s=4.0,    # Max decay rate
                         min_series_peak_dff=0.25,     # Min peak in event series
                         series_timeout_s=5.0,         # Time limit for series peak
                         merge_gap_s=0.3,              # Merge events within gap
                         min_snr=1.2,                  # Minimum signal-to-noise ratio
                         # Shape-based filtering
                         min_event_duration_s=0.05,    # Minimum event duration
                         max_baseline_drift_per_s=0.5, # Max allowed baseline drift during event
                         require_clear_peak=True):     # Require obvious peak structure
    """
    Enhanced Ca²⁺ transient detection using all config parameters
    
    Returns:
      events: list of dicts with {t_on, t_peak, t_off, amp, rise_s, decay_s, area}
      mask:   boolean array True during accepted events
    """
    x = np.asarray(x, float)
    T = x.size

    if T < 2:
        return [], np.zeros(T, dtype=bool)
    
    # Check for invalid traces
    if np.all(np.isnan(x)) or np.all(x == x[0]):
        return [], np.zeros(T, dtype=bool)

    # Light smoothing
    if smooth_sigma_s and smooth_sigma_s > 0:
        sigma_frames = max(1, int(round(smooth_sigma_s * fs)))
        x_s = gaussian_filter1d(x, sigma=sigma_frames)
    else:
        x_s = x.copy()

    # Robust noise estimation
    try:
        valid_mask = np.isfinite(x_s)
        if not np.any(valid_mask):
            return [], np.zeros(T, dtype=bool)
        
        x_valid = x_s[valid_mask]
        
        if len(x_valid) < 10 or np.std(x_valid) < 1e-12:
            return [], np.zeros(T, dtype=bool)
        
        # Use multiple percentiles for more robust baseline estimation
        p10 = np.percentile(x_valid, 10)
        p20 = np.percentile(x_valid, 20) 
        p30 = np.percentile(x_valid, 30)
        
        # Use the most stable baseline region
        baseline_candidates = [p10, p20, p30]
        residuals = []
        for p in baseline_candidates:
            r = x_valid - p
            mad = np.median(np.abs(r - np.median(r)))
            residuals.append(mad)
        
        # Choose baseline with smallest MAD (most stable)
        best_baseline = baseline_candidates[np.argmin(residuals)]
        r = x_valid - best_baseline
        mad = np.median(np.abs(r - np.median(r)))
        
        if mad < 1e-12:
            mad = np.std(x_valid) / 1.4826
            if mad < 1e-12:
                return [], np.zeros(T, dtype=bool)
        
        sigma = 1.4826 * mad
        
    except (ValueError, RuntimeError):
        return [], np.zeros(T, dtype=bool)

    # Adaptive thresholds based on signal characteristics
    try:
        if use_z:
            med_val = np.median(x_s[valid_mask])
            z = (x_s - med_val) / (sigma + 1e-12)
            th_on, th_off = k_on, k_off
            y = z
            amp_thr = 0.0
        else:
            # Use the robust baseline estimate
            med_val = best_baseline
            y = x_s
            th_on = med_val + k_on * sigma
            th_off = med_val + k_off * sigma
            amp_thr = min_amp
            
    except (ValueError, RuntimeError):
        return [], np.zeros(T, dtype=bool)

    # Convert time parameters to samples
    rise_max = int(round(rise_max_s * fs))
    peak_win = int(round(peak_win_s * fs))
    decay_max = int(round(decay_max_s * fs))
    series_timeout = int(round(series_timeout_s * fs))
    merge_gap = int(round(merge_gap_s * fs))
    min_duration = int(round(min_event_duration_s * fs))

    events = []
    mask = np.zeros(T, dtype=bool)
    i = 1
    
    try:
        while i < T:
            # Skip NaN values
            if not np.isfinite(y[i-1]) or not np.isfinite(y[i]):
                i += 1
                continue
                
            # ONSET: upward crossing of th_on
            if y[i-1] < th_on and y[i] >= th_on:
                i_on = i

                # Find baseline (last crossing below th_off)
                j = i_on
                while j > 0 and np.isfinite(y[j-1]) and y[j-1] > th_off:
                    j -= 1
                i_base = j
                
                if not np.isfinite(y[i_base]):
                    i += 1
                    continue
                    
                baseline = y[i_base]

                # Rise time check
                j = i_on
                while j > 0 and np.isfinite(y[j-1]) and y[j-1] > th_off:
                    j -= 1
                rise_len = i_on - j
                if rise_len > rise_max:
                    i += 1
                    continue

                # Peak search in event series
                i_end_search = min(T, i_on + series_timeout)
                max_peak_in_series = np.nanmax(y[i_on:i_end_search]) if i_end_search > i_on else y[i_on]
                series_amp = max_peak_in_series - baseline
                
                if series_amp < min_series_peak_dff:
                    i += 1
                    continue

                # Find peak within initial window
                i_end_peak = min(T, i_on + max(peak_win, 1))
                peak_segment = y[i_on:i_end_peak]
                valid_peak_mask = np.isfinite(peak_segment)
                if not np.any(valid_peak_mask):
                    i += 1
                    continue
                
                local_peak_idx = np.nanargmax(peak_segment)
                i_peak = i_on + local_peak_idx
                peak_val = y[i_peak]
                
                if not np.isfinite(peak_val):
                    i += 1
                    continue
                
                amp = peak_val - baseline
                if amp < amp_thr:
                    i += 1
                    continue

                # Find decay with merging allowed
                i_search_end = min(T, i_on + max(decay_max, 1))
                k = i_peak
                i_off = None
                
                while k < i_search_end:
                    if not np.isfinite(y[k]):
                        k += 1
                        continue
                        
                    # Allow re-crossings above onset (complex events)
                    if y[k] >= th_on and y[k] > peak_val:
                        i_peak, peak_val = k, y[k]
                        amp = peak_val - baseline
                    
                    if y[k] < th_off:
                        i_off = k
                        break
                    k += 1

                if i_off is None:
                    i += 1
                    continue

                # Event duration check
                event_duration_s = (i_off - i_base) / fs
                if event_duration_s < min_event_duration_s:
                    i += 1
                    continue

                # Decay slope constraint
                decay_s = (i_off - i_peak) / fs
                if decay_s > 0:
                    decay_slope = (peak_val - y[i_off]) / decay_s
                    if decay_slope > max_decay_slope_per_s:
                        i += 1
                        continue

                # Baseline stability check during event
                if max_baseline_drift_per_s > 0:
                    event_segment = y[i_base:i_off]
                    if len(event_segment) > 3:
                        # Check if baseline drifts too much during event
                        baseline_trend = np.polyfit(np.arange(len(event_segment)), event_segment, 1)[0]
                        baseline_drift_rate = abs(baseline_trend * fs)
                        if baseline_drift_rate > max_baseline_drift_per_s:
                            i += 1
                            continue

                # Peak prominence check (clear peak requirement)
                if require_clear_peak and len(event_segment) > 5:
                    # Require peak to be clearly above surrounding region
                    peak_rel_idx = i_peak - i_base
                    if peak_rel_idx > 2 and peak_rel_idx < len(event_segment) - 2:
                        surrounding = np.concatenate([
                            event_segment[max(0, peak_rel_idx-2):peak_rel_idx],
                            event_segment[peak_rel_idx+1:min(len(event_segment), peak_rel_idx+3)]
                        ])
                        if len(surrounding) > 0:
                            peak_prominence = peak_val - np.max(surrounding)
                            if peak_prominence < 0.3 * amp:  # Peak must be 30% above surroundings
                                i += 1
                                continue

                # SNR requirement
                if min_snr > 0:
                    pre_event_start = max(0, i_base - 30)  # Shorter pre-event window
                    if i_base > pre_event_start:
                        noise_segment = y[pre_event_start:i_base]
                        noise_std = np.nanstd(noise_segment[np.isfinite(noise_segment)])
                        if noise_std > 0:
                            local_snr = amp / noise_std
                            if local_snr < min_snr:
                                i += 1
                                continue

                # Calculate event metrics
                rise_s = (i_peak - i_base) / fs
                decay_s = (i_off - i_peak) / fs
                
                # Calculate area
                event_segment = y[i_base:i_off]
                valid_event_mask = np.isfinite(event_segment)
                if np.any(valid_event_mask):
                    area_values = np.maximum(event_segment[valid_event_mask] - baseline, 0)
                    area = np.trapz(area_values, dx=1/fs)
                else:
                    area = 0.0

                # Event merging
                if events and (i_on - events[-1]['i_off']) <= merge_gap:
                    # Merge with previous event
                    last_event = events[-1]
                    
                    # Update peak if current is higher
                    if peak_val > y[last_event['i_peak']]:
                        events[-1].update({
                            't_peak': i_peak/fs,
                            'i_peak': i_peak,
                            'amp': max(amp, last_event['amp'])
                        })
                    
                    # Extend off time
                    events[-1]['t_off'] = i_off/fs
                    events[-1]['i_off'] = i_off
                    events[-1]['decay_s'] = (i_off - events[-1]['i_peak']) / fs
                    
                    # Recalculate area
                    merged_segment = y[last_event['i_base']:i_off]
                    valid_merged_mask = np.isfinite(merged_segment)
                    if np.any(valid_merged_mask):
                        merged_area_values = np.maximum(merged_segment[valid_merged_mask] - last_event['baseline'], 0)
                        events[-1]['area'] = np.trapz(merged_area_values, dx=1/fs)
                    
                    # Update mask
                    mask[last_event['i_base']:i_off] = True
                    
                else:
                    # Create new event
                    events.append(dict(
                        t_on=i_base/fs, t_peak=i_peak/fs, t_off=i_off/fs,
                        amp=amp, rise_s=rise_s, decay_s=decay_s, area=area,
                        i_on=i_on, i_peak=i_peak, i_off=i_off, baseline=baseline,
                        i_base=i_base
                    ))
                    mask[i_base:i_off] = True

                i = i_off + 1
            else:
                i += 1

    except Exception:
        # Return partial results if something went wrong
        pass

    return events, mask


# Update compute_qc_metrics to use the updated detect_ca_transients function
def compute_qc_metrics(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """STEP 4C: Compute QC metrics including spike rates, SNR, and Ca transients with updated detection"""
    print("\n=== COMPUTING QC METRICS ===")
    
    # Check for spike data
    spike_version = None
    if 'spks_oasis' in data:
        spike_version = 'spks_oasis'
    elif 'spks_oasis_recon' in data:
        spike_version = 'spks_oasis_recon'
    
    if spike_version is None:
        print("No spike data available - skipping spike-based QC")
        
    fs = float(cfg["acq"]["fs"])
    
    # Get dF/F data for Ca transient detection
    dff_clean = data.get('dFF_clean')
    if dff_clean is None:
        print("No dF/F data available - skipping Ca transient detection")
        return
    
    N, T = dff_clean.shape
    print(f"Computing QC metrics for {N} ROIs x {T} frames at {fs:.1f} Hz")
    
    # Updated Ca transient detection parameters - use ALL available config parameters
    ca_cfg = cfg.get("ca_transients", {})
    
    # Extract all parameters that the updated function expects
    detection_params = {
        'smooth_sigma_s': ca_cfg.get('smooth_sigma_s', 0.10),
        'k_on': ca_cfg.get('k_on', 2.0),
        'k_off': ca_cfg.get('k_off', 0.8),
        'rise_max_s': ca_cfg.get('rise_max_s', 0.50),
        'peak_win_s': ca_cfg.get('peak_win_s', 1.50),
        'decay_max_s': ca_cfg.get('decay_max_s', 4.0),
        'min_amp': ca_cfg.get('min_amp', 0.15),
        'use_z': ca_cfg.get('use_z', False),
        'max_decay_slope_per_s': ca_cfg.get('max_decay_slope_per_s', 4.0),
        'min_series_peak_dff': ca_cfg.get('min_series_peak_dff', 0.25),
        'series_timeout_s': ca_cfg.get('series_timeout_s', 5.0),
        'merge_gap_s': ca_cfg.get('merge_gap_s', 0.3),
        'min_snr': ca_cfg.get('min_snr', 1.2),
        'min_event_duration_s': ca_cfg.get('min_event_duration_s', 0.05),
        'max_baseline_drift_per_s': ca_cfg.get('max_baseline_drift_per_s', 0.5),
        'require_clear_peak': ca_cfg.get('require_clear_peak', True)
    }
    
    print(f"Updated Ca transient detection parameters:")
    print(f"  Basic: smoothing={detection_params['smooth_sigma_s']}s, thresholds={detection_params['k_on']}σ/{detection_params['k_off']}σ")
    print(f"  Amplitude: min_amp={detection_params['min_amp']}, min_series_peak={detection_params['min_series_peak_dff']}")
    print(f"  Kinetics: rise_max={detection_params['rise_max_s']}s, decay_max={detection_params['decay_max_s']}s")
    print(f"  Quality: min_snr={detection_params['min_snr']}, min_duration={detection_params['min_event_duration_s']}s")
    
    # Detect Ca transients for each ROI with updated algorithm
    ca_events_list = []
    ca_masks_list = []
    ca_event_counts = np.zeros(N, dtype=np.int32)
    ca_event_rates = np.zeros(N, dtype=np.float32)
    ca_mean_amplitudes = np.zeros(N, dtype=np.float32)
    ca_mean_rise_times = np.zeros(N, dtype=np.float32)
    ca_mean_decay_times = np.zeros(N, dtype=np.float32)
    ca_total_active_time = np.zeros(N, dtype=np.float32)
    
    print("  Detecting Ca transients with updated algorithm...")
    for i in range(N):
        if i % 50 == 0 and i > 0:
            print(f"    Processed {i}/{N} ROIs...")
        
        try:
            # Use the updated detect_ca_transients function with all parameters
            events, mask = detect_ca_transients(dff_clean[i], fs, **detection_params)
            
            ca_events_list.append(events)
            ca_masks_list.append(mask)
            
            # Summary statistics
            ca_event_counts[i] = len(events)
            ca_event_rates[i] = len(events) / (T / fs)  # events per second
            ca_total_active_time[i] = np.sum(mask) / fs  # seconds of active time
            
            if len(events) > 0:
                ca_mean_amplitudes[i] = np.mean([e['amp'] for e in events])
                ca_mean_rise_times[i] = np.mean([e['rise_s'] for e in events])
                ca_mean_decay_times[i] = np.mean([e['decay_s'] for e in events])
            else:
                ca_mean_amplitudes[i] = 0.0
                ca_mean_rise_times[i] = 0.0
                ca_mean_decay_times[i] = 0.0
                
        except Exception as e:
            print(f"    Warning: Updated Ca transient detection failed for ROI {i}: {e}")
            ca_events_list.append([])
            ca_masks_list.append(np.zeros(T, dtype=bool))
    
    # Process spike data if available
    if spike_version is not None:
        spikes = data[spike_version]
        
        # Parameters for spike detection and binning
        qc_cfg = cfg.get("qc_metrics", {})
        k_mad = qc_cfg.get("onset_threshold_mad", 3.0)
        bin_duration_s = qc_cfg.get("spike_rate_bin_s", 0.2)
        
        print(f"  Computing spike rates from {spike_version}")
        print(f"    Onset detection threshold: {k_mad} MAD")
        print(f"    Spike rate binning: {bin_duration_s}s")
        
        # Spike detection and rates
        spike_onsets = spike_onset_mask(spikes, k_mad=k_mad)
        spike_rate_hz_binned = spike_rate_hz(spikes, fs, bin_duration_s, k_mad=k_mad)
        spike_rate_hz_instantaneous = spike_onsets.astype(np.float32) * fs
        
        # Smoothed spike rates for plotting
        from scipy.ndimage import gaussian_filter1d
        smooth_sigma_frames = max(1.0, 0.1 * fs)
        spike_rate_smoothed_hz = np.array([gaussian_filter1d(spike_rate_hz_instantaneous[i], 
                                                             sigma=smooth_sigma_frames) 
                                          for i in range(N)])
        
        # Spike statistics
        spike_metrics = {
            'spike_raw': spikes,
            'spike_onsets': spike_onsets,
            'spike_rate_hz_instantaneous': spike_rate_hz_instantaneous,
            'spike_rate_hz_smoothed': spike_rate_smoothed_hz,
            'spike_count_onsets': np.sum(spike_onsets, axis=1),
            'spike_rate_mean_hz': np.mean(spike_rate_hz_instantaneous, axis=1),
            'spike_amplitude_max': np.max(spikes, axis=1),
            'spike_cv_isi_onsets': compute_spike_cv_isi(spike_onsets, fs),
        }
    else:
        spike_metrics = {}
    
    # Combined QC metrics with updated Ca transient data
    qc_metrics = {
        # Updated Ca transient detection
        'ca_events': ca_events_list,  # List of event dicts per ROI
        'ca_masks': ca_masks_list,    # List of boolean masks per ROI
        'ca_event_counts': ca_event_counts,
        'ca_event_rates_hz': ca_event_rates,
        'ca_mean_amplitudes': ca_mean_amplitudes,
        'ca_mean_rise_times_s': ca_mean_rise_times,
        'ca_mean_decay_times_s': ca_mean_decay_times,
        'ca_total_active_time_s': ca_total_active_time,
        'ca_activity_fraction': ca_total_active_time / (T / fs),  # fraction of time active
        
        # Spike metrics (if available)
        **spike_metrics,
        
        # Signal quality
        'snr_estimate': estimate_signal_noise_ratio(data),
        'temporal_stability': compute_temporal_stability(data),
        
        # Metadata with updated parameters
        'ca_detection_params': detection_params,
        'spike_version': spike_version,
    }
    
    data['qc_metrics'] = qc_metrics
    
    # Updated summary statistics
    total_ca_events = ca_event_counts.sum()
    mean_ca_rate = ca_event_rates.mean()
    mean_ca_amplitude = ca_mean_amplitudes[ca_event_counts > 0].mean() if np.any(ca_event_counts > 0) else 0.0
    mean_activity_fraction = qc_metrics['ca_activity_fraction'].mean()
    
    print(f"Updated Ca transient detection results:")
    print(f"  Total Ca events detected: {total_ca_events}")
    print(f"  Mean Ca event rate: {mean_ca_rate:.3f} events/s")
    print(f"  Mean Ca amplitude: {mean_ca_amplitude:.3f} ΔF/F")
    print(f"  Mean activity fraction: {100*mean_activity_fraction:.1f}%")
    print(f"  ROIs with events: {np.sum(ca_event_counts > 0)}/{N} ({100*np.sum(ca_event_counts > 0)/N:.1f}%)")
    
    if spike_version is not None:
        max_spike_rate = spike_metrics['spike_rate_mean_hz'].max()
        print(f"Spike detection results:")
        print(f"  Max spike rate: {max_spike_rate:.1f} Hz")
        print(f"  Total spike onsets: {spike_metrics['spike_count_onsets'].sum()}")
    
    print("Updated QC metrics computation complete")


# def plot_roi_review(data: Dict[str, Any], cfg: Dict[str, Any], roi: int, save: bool = False) -> Any:
#     """Plot detailed ROI review figure with spike and spike rate subplots"""
#     print(f"\n=== PLOTTING ROI {roi} REVIEW ===")
    
#     if "dFF" not in data or "Fc" not in data or "F0" not in data:
#         raise ValueError("Missing processed arrays - run earlier steps")
    
#     rv = cfg['review']
#     segments = list(rv['detail_segments'])
#     fig_w = float(rv['fig_width'])
#     row_h = float(rv['row_height'])
#     dpi = int(rv['dpi'])
#     fs = float(cfg['acq']['fs'])
    
#     N, T = data["dFF"].shape
#     if roi < 0 or roi >= N:
#         raise IndexError(f"ROI {roi} out of range (N={N})")
    
#     if 'review' not in data:
#         raise ValueError("Review cache missing - call prepare_review_cache first")
    
#     rev = data['review']
#     if rev.get('T') != T:
#         raise ValueError("Review cache stale - re-run prepare_review_cache")
    
#     # Check for cached spike data
#     has_spikes = rev.get('spks_ds') is not None
#     spike_version = rev.get('spike_version')
    
#     if has_spikes:
#         print(f"  Using cached spike data: {spike_version}")
#     else:
#         print(f"  No spike data available in cache")
    
#     detail_idx = _segments_to_idx(segments, fs, T)
#     n_detail = len(detail_idx)
    
#     # Calculate rows: 2 full traces + detail segments
#     base_rows = 2  # F/Fneu + Fc/F0
    
#     if has_spikes:
#         detail_rows = 5 * n_detail  # Fc/F0 + spikes + spike_rate + dFF + dFF_clean
#     else:
#         detail_rows = 3 * n_detail  # Fc/F0 + dFF + dFF_clean
    
#     rows = base_rows + detail_rows
#     fig_h = max(2.5, rows * row_h)
    
#     print(f"Creating figure: {rows} rows, {n_detail} detail segments, spikes: {has_spikes}")
    
#     fig, axes = plt.subplots(rows, 1, figsize=(fig_w, fig_h), sharex=False)
#     if rows == 1:
#         axes = [axes]
    
#     r = 0
    
#     # Full traces
#     ax = axes[r]; r += 1
#     ax.plot(rev['t_ds'], rev['F_ds'][roi], lw=0.4, color='k', label='F')
#     if rev.get('Fneu_ds') is not None:
#         ax.plot(rev['t_ds'], rev['Fneu_ds'][roi], lw=0.4, color='magenta', alpha=0.6, label='Fneu')
#     ax.set_ylabel('F')
#     ax.set_title(f'ROI {roi} full F/Fneu')
#     ax.legend(frameon=False, fontsize=7, loc='upper right')
#     # Add horizontal line at 0 if within range
#     ylim = ax.get_ylim()
#     if ylim[0] <= 0 <= ylim[1]:
#         ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    
#     ax = axes[r]; r += 1
#     ax.plot(rev['t_ds'], rev['Fc_ds'][roi], lw=0.4, color='steelblue', label='Fc')
#     ax.plot(rev['t_ds'], rev['F0_ds'][roi], lw=0.5, color='orange', label='F0')
#     ax.set_ylabel('Fc')
#     ax.set_title('Full Fc & F0')
#     ax.legend(frameon=False, fontsize=7, loc='upper right')
#     # Add horizontal line at 0 if within range
#     ylim = ax.get_ylim()
#     if ylim[0] <= 0 <= ylim[1]:
#         ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    
#     # Detail segments
#     for seg_idx, (slc, t_local, (sa, sb)) in enumerate(detail_idx):
#         # 1. Fc + F0
#         ax = axes[r]; r += 1
#         ax.plot(t_local, data['Fc'][roi, slc], lw=0.5, color='steelblue', label='Fc')
#         ax.plot(t_local, data['F0'][roi, slc], lw=0.5, color='orange', label='F0')
#         ax.set_ylabel('Fc')
#         ax.set_title(f'{sa:.1f}-{sb:.1f}s Fc/F0')
#         ax.legend(frameon=False, fontsize=7, loc='upper right')
#         ax.set_xlim(sa, sb)
#         # Add horizontal line at 0 if within range
#         ylim = ax.get_ylim()
#         if ylim[0] <= 0 <= ylim[1]:
#             ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        
#         if has_spikes:
#             # Get full resolution spike data for this segment
#             full_spike_data = data.get(spike_version)
#             if full_spike_data is not None:
#                 spikes_detail = full_spike_data[roi, slc]
                
#                 # 2. Spikes (raw OASIS deconvolution values)
#                 ax = axes[r]; r += 1
#                 ax.plot(t_local, spikes_detail, lw=0.6, color='red')
                
#                 # Mark events
#                 if np.max(full_spike_data[roi]) > 0:
#                     spike_threshold = 0.1 * np.max(full_spike_data[roi])
#                     event_mask = spikes_detail > spike_threshold
                    
#                     if np.any(event_mask):
#                         ax.scatter(t_local[event_mask], spikes_detail[event_mask], 
#                                   c='darkred', s=12, alpha=0.9, zorder=5)
                        
#                         # Light background shading for events
#                         event_times = t_local[event_mask]
#                         for et in event_times:
#                             ax.axvspan(et-0.05, et+0.05, alpha=0.15, color='red', zorder=1)
                        
#                         n_events = np.sum(event_mask)
#                         ax.set_title(f'Spikes ({n_events} events)')
#                     else:
#                         ax.set_title('Spikes (no events)')
                    
#                     if np.max(spikes_detail) > 0:
#                         ax.set_ylim(0, np.max(spikes_detail) * 1.1)
#                 else:
#                     ax.set_title('Spikes (no data)')
                
#                 ax.set_ylabel('Spikes')
#                 ax.set_xlim(sa, sb)
#                 # Add horizontal line at 0 if within range
#                 ylim = ax.get_ylim()
#                 if ylim[0] <= 0 <= ylim[1]:
#                     ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
                
#                 # 3. Spike Rate (converted to Hz)
#                 ax = axes[r]; r += 1
                
#                 qc = data.get('qc_metrics', {})
#                 if 'spike_rate_hz_smoothed' in qc:
#                     # Use the smoothed onset-based rate for visual consistency
#                     spike_rate_detail = qc['spike_rate_hz_smoothed'][roi, slc]
#                     method_label = "onset-based (smoothed)"
#                 elif 'spike_rate_hz_instantaneous' in qc:
#                     # Fallback to instantaneous onset rate
#                     spike_rate_detail = qc['spike_rate_hz_instantaneous'][roi, slc]
#                     method_label = "instantaneous (onsets)"
#                 else:
#                     # Final fallback - should not happen with fix
#                     spike_rate_detail = np.zeros_like(t_local)
#                     method_label = "no data"
                
#                 ax.plot(t_local, spike_rate_detail, lw=0.6, color='darkred', alpha=0.8)
                
#                 # Title and labels
#                 max_rate = np.max(spike_rate_detail) if len(spike_rate_detail) > 0 else 0
#                 ax.set_title(f'Spike Rate ({method_label}, max: {max_rate:.1f} Hz)')
#                 ax.set_ylabel('Spike Rate (Hz)')
#                 ax.set_xlim(sa, sb)
                
#                 # Add horizontal line at 0
#                 ylim = ax.get_ylim()
#                 if ylim[0] <= 0 <= ylim[1]:
#                     ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
#             else:
#                 # No spike data available - create empty subplots
#                 ax = axes[r]; r += 1  # Spikes
#                 ax.set_title('Spikes (data missing)')
#                 ax.set_ylabel('Spikes')
#                 ax.set_xlim(sa, sb)
                
#                 ax = axes[r]; r += 1  # Spike Rate
#                 ax.set_title('Spike Rate (data missing)')
#                 ax.set_ylabel('Spike Rate (Hz)')
#                 ax.set_xlim(sa, sb)
        
#         # 4. dFF with Ca transient shading
#         ax = axes[r]; r += 1
#         ax.plot(t_local, data['dFF'][roi, slc], lw=0.5, color='teal')
        
#         # Add Ca transient shading if available
#         qc = data.get('qc_metrics', {})
#         if 'ca_masks' in qc and len(qc['ca_masks']) > roi:
#             ca_mask_full = qc['ca_masks'][roi]
#             ca_mask_segment = ca_mask_full[slc]
            
#             if np.any(ca_mask_segment):
#                 # Create shading for Ca transient periods
#                 ca_times = t_local[ca_mask_segment]
#                 ca_values = data['dFF'][roi, slc][ca_mask_segment]
                
#                 # Fill between baseline and signal during Ca events
#                 baseline = np.min(data['dFF'][roi, slc])
#                 ax.fill_between(t_local, baseline, data['dFF'][roi, slc], 
#                                where=ca_mask_segment, alpha=0.3, color='orange', 
#                                label='Ca events', interpolate=True)
                
#                 # Count events in this segment
#                 if 'ca_events' in qc and len(qc['ca_events']) > roi:
#                     events_in_segment = [e for e in qc['ca_events'][roi] 
#                                        if sa <= e['t_peak'] <= sb]
#                     n_events = len(events_in_segment)
                    
#                     # Mark event peaks
#                     for event in events_in_segment:
#                         peak_idx = int(round((event['t_peak'] - sa) * fs))
#                         if 0 <= peak_idx < len(t_local):
#                             ax.scatter(event['t_peak'], data['dFF'][roi, slc][peak_idx], 
#                                      c='red', s=20, marker='^', zorder=5)
                    
#                     ax.set_title(f'dF/F (raw) - {n_events} Ca events')
#                 else:
#                     ax.set_title('dF/F (raw) - Ca events detected')
#             else:
#                 ax.set_title('dF/F (raw) - no Ca events')
#         else:
#             ax.set_title('dF/F (raw)')
        
#         ax.set_ylabel('dFF')
#         ax.set_xlim(sa, sb)
#         # Add horizontal line at 0 if within range
#         ylim = ax.get_ylim()
#         if ylim[0] <= 0 <= ylim[1]:
#             ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        
#         # 5. dFF clean with Ca transient shading
#         ax = axes[r]; r += 1
#         ax.plot(t_local, data['dFF_clean'][roi, slc], lw=0.5, color='purple')
        
#         # Add Ca transient shading for clean trace too
#         if 'ca_masks' in qc and len(qc['ca_masks']) > roi:
#             ca_mask_full = qc['ca_masks'][roi]
#             ca_mask_segment = ca_mask_full[slc]
            
#             if np.any(ca_mask_segment):
#                 baseline = np.min(data['dFF_clean'][roi, slc])
#                 ax.fill_between(t_local, baseline, data['dFF_clean'][roi, slc], 
#                                where=ca_mask_segment, alpha=0.3, color='orange', 
#                                label='Ca events', interpolate=True)
                
#                 # Event statistics for clean trace
#                 if 'ca_events' in qc and len(qc['ca_events']) > roi:
#                     events_in_segment = [e for e in qc['ca_events'][roi] 
#                                        if sa <= e['t_peak'] <= sb]
#                     total_amp = sum(e['amp'] for e in events_in_segment)
#                     ax.set_title(f'dF/F (clean) - {len(events_in_segment)} events, Σamp={total_amp:.2f}')
#                 else:
#                     ax.set_title('dF/F (clean) - Ca events detected')
#             else:
#                 ax.set_title('dF/F (clean) - no Ca events')
#         else:
#             ax.set_title('dF/F (clean)')
        
#         ax.set_ylabel('dFF*')
#         ax.set_xlim(sa, sb)
#         # Add horizontal line at 0 if within range
#         ylim = ax.get_ylim()
#         if ylim[0] <= 0 <= ylim[1]:
#             ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)

        
#         # # 5. dFF clean
#         # ax = axes[r]; r += 1
#         # ax.plot(t_local, data['dFF_clean'][roi, slc], lw=0.5, color='purple')
#         # ax.set_ylabel('dFF*')
#         # ax.set_title('dF/F (clean)')
#         # ax.set_xlim(sa, sb)
#         # # Add horizontal line at 0 if within range
#         # ylim = ax.get_ylim()
#         # if ylim[0] <= 0 <= ylim[1]:
#         #     ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    
#     axes[-1].set_xlabel('Time (s)')
#     fig.tight_layout()
    
#     if save:
#         out_dir = rv['save_dir']
#         os.makedirs(out_dir, exist_ok=True)
#         path = os.path.join(out_dir, f"roi_{roi:04d}.png")
#         fig.savefig(path, dpi=dpi)
#         print(f"Review figure saved: {path}")
#     else:
#         print(f"Review figure displayed")
    
#     return fig

def plot_roi_review(data: Dict[str, Any], cfg: Dict[str, Any], roi: int, save: bool = False) -> Any:
    """Plot detailed ROI review figure with spikes, spike rate, and enhanced Ca²⁺ transient visualization"""
    print(f"\n=== PLOTTING ROI {roi} REVIEW ===")
    
    if "dFF" not in data or "Fc" not in data or "F0" not in data:
        raise ValueError("Missing processed arrays - run earlier steps")
    
    rv = cfg['review']
    segments = list(rv['detail_segments'])
    fig_w = float(rv['fig_width'])
    row_h = float(rv['row_height'])
    dpi = int(rv['dpi'])
    fs = float(cfg['acq']['fs'])
    
    N, T = data["dFF"].shape
    if roi < 0 or roi >= N:
        raise IndexError(f"ROI {roi} out of range (N={N})")
    
    if 'review' not in data:
        raise ValueError("Review cache missing - call prepare_review_cache first")
    
    rev = data['review']
    if rev.get('T') != T:
        raise ValueError("Review cache stale - re-run prepare_review_cache")
    
    # Check for cached spike data
    has_spikes = rev.get('spks_ds') is not None
    spike_version = rev.get('spike_version')
    
    if has_spikes:
        print(f"  Using cached spike data: {spike_version}")
    else:
        print(f"  No spike data available in cache")
    
    detail_idx = _segments_to_idx(segments, fs, T)
    n_detail = len(detail_idx)
    
    # Calculate rows: 2 full traces + detail segments
    base_rows = 2  # F/Fneu + Fc/F0
    
    if has_spikes:
        detail_rows = 5 * n_detail  # Fc/F0 + spikes + spike_rate + dFF + dFF_clean
    else:
        detail_rows = 3 * n_detail  # Fc/F0 + dFF + dFF_clean
    
    rows = base_rows + detail_rows
    fig_h = max(2.5, rows * row_h)
    
    print(f"Creating figure: {rows} rows, {n_detail} detail segments, spikes: {has_spikes}")
    
    fig, axes = plt.subplots(rows, 1, figsize=(fig_w, fig_h), sharex=False)
    if rows == 1:
        axes = [axes]
    
    r = 0
    
    # Full traces
    ax = axes[r]; r += 1
    ax.plot(rev['t_ds'], rev['F_ds'][roi], lw=0.4, color='k', label='F')
    if rev.get('Fneu_ds') is not None:
        ax.plot(rev['t_ds'], rev['Fneu_ds'][roi], lw=0.4, color='magenta', alpha=0.6, label='Fneu')
    ax.set_ylabel('F')
    ax.set_title(f'ROI {roi} full F/Fneu')
    ax.legend(frameon=False, fontsize=7, loc='upper right')
    # Add horizontal line at 0 if within range
    ylim = ax.get_ylim()
    if ylim[0] <= 0 <= ylim[1]:
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    
    ax = axes[r]; r += 1
    ax.plot(rev['t_ds'], rev['Fc_ds'][roi], lw=0.4, color='steelblue', label='Fc')
    ax.plot(rev['t_ds'], rev['F0_ds'][roi], lw=0.5, color='orange', label='F0')
    ax.set_ylabel('Fc')
    ax.set_title('Full Fc & F0')
    ax.legend(frameon=False, fontsize=7, loc='upper right')
    # Add horizontal line at 0 if within range
    ylim = ax.get_ylim()
    if ylim[0] <= 0 <= ylim[1]:
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # Detail segments
    for seg_idx, (slc, t_local, (sa, sb)) in enumerate(detail_idx):
        # 1. Fc + F0
        ax = axes[r]; r += 1
        ax.plot(t_local, data['Fc'][roi, slc], lw=0.5, color='steelblue', label='Fc')
        ax.plot(t_local, data['F0'][roi, slc], lw=0.5, color='orange', label='F0')
        ax.set_ylabel('Fc')
        ax.set_title(f'{sa:.1f}-{sb:.1f}s Fc/F0')
        ax.legend(frameon=False, fontsize=7, loc='upper right')
        ax.set_xlim(sa, sb)
        # Add horizontal line at 0 if within range
        ylim = ax.get_ylim()
        if ylim[0] <= 0 <= ylim[1]:
            ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        
        if has_spikes:
            # Get full resolution spike data for this segment
            full_spike_data = data.get(spike_version)
            if full_spike_data is not None:
                spikes_detail = full_spike_data[roi, slc]
                
                # 2. Spikes (raw OASIS deconvolution values)
                ax = axes[r]; r += 1
                ax.plot(t_local, spikes_detail, lw=0.6, color='red')
                
                # Mark events
                if np.max(full_spike_data[roi]) > 0:
                    spike_threshold = 0.1 * np.max(full_spike_data[roi])
                    event_mask = spikes_detail > spike_threshold
                    
                    if np.any(event_mask):
                        ax.scatter(t_local[event_mask], spikes_detail[event_mask], 
                                  c='darkred', s=12, alpha=0.9, zorder=5)
                        
                        # Light background shading for events
                        event_times = t_local[event_mask]
                        for et in event_times:
                            ax.axvspan(et-0.05, et+0.05, alpha=0.15, color='red', zorder=1)
                        
                        n_events = np.sum(event_mask)
                        ax.set_title(f'Spikes ({n_events} events)')
                    else:
                        ax.set_title('Spikes (no events)')
                    
                    if np.max(spikes_detail) > 0:
                        ax.set_ylim(0, np.max(spikes_detail) * 1.1)
                else:
                    ax.set_title('Spikes (no data)')
                
                ax.set_ylabel('Spikes')
                ax.set_xlim(sa, sb)
                # Add horizontal line at 0 if within range
                ylim = ax.get_ylim()
                if ylim[0] <= 0 <= ylim[1]:
                    ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
                
                # 3. Spike Rate (converted to Hz)
                ax = axes[r]; r += 1
                
                qc = data.get('qc_metrics', {})
                if 'spike_rate_hz_smoothed' in qc:
                    # Use the smoothed onset-based rate for visual consistency
                    spike_rate_detail = qc['spike_rate_hz_smoothed'][roi, slc]
                    method_label = "onset-based (smoothed)"
                elif 'spike_rate_hz_instantaneous' in qc:
                    # Fallback to instantaneous onset rate
                    spike_rate_detail = qc['spike_rate_hz_instantaneous'][roi, slc]
                    method_label = "instantaneous (onsets)"
                else:
                    # Final fallback - should not happen with fix
                    spike_rate_detail = np.zeros_like(t_local)
                    method_label = "no data"
                
                ax.plot(t_local, spike_rate_detail, lw=0.6, color='darkred', alpha=0.8)
                
                # Title and labels
                max_rate = np.max(spike_rate_detail) if len(spike_rate_detail) > 0 else 0
                ax.set_title(f'Spike Rate ({method_label}, max: {max_rate:.1f} Hz)')
                ax.set_ylabel('Spike Rate (Hz)')
                ax.set_xlim(sa, sb)
                
                # Add horizontal line at 0
                ylim = ax.get_ylim()
                if ylim[0] <= 0 <= ylim[1]:
                    ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
            else:
                # No spike data available - create empty subplots
                ax = axes[r]; r += 1  # Spikes
                ax.set_title('Spikes (data missing)')
                ax.set_ylabel('Spikes')
                ax.set_xlim(sa, sb)
                
                ax = axes[r]; r += 1  # Spike Rate
                ax.set_title('Spike Rate (data missing)')
                ax.set_ylabel('Spike Rate (Hz)')
                ax.set_xlim(sa, sb)
        
        # 4. dFF with enhanced Ca²⁺ transient visualization
        ax = axes[r]; r += 1
        ax.plot(t_local, data['dFF'][roi, slc], lw=0.5, color='teal')
        
        # Add enhanced Ca transient shading if available
        qc = data.get('qc_metrics', {})
        if 'ca_masks' in qc and len(qc['ca_masks']) > roi:
            ca_mask_full = qc['ca_masks'][roi]
            ca_mask_segment = ca_mask_full[slc]
            
            if np.any(ca_mask_segment):
                # Create shading for Ca transient periods
                baseline = np.min(data['dFF'][roi, slc])
                ax.fill_between(t_local, baseline, data['dFF'][roi, slc], 
                               where=ca_mask_segment, alpha=0.3, color='orange', 
                               label='Ca events', interpolate=True)
                
                # Enhanced event marking and statistics
                if 'ca_events' in qc and len(qc['ca_events']) > roi:
                    events_in_segment = [e for e in qc['ca_events'][roi] 
                                       if sa <= e['t_peak'] <= sb]
                    n_events = len(events_in_segment)
                    
                    # Mark event features with different markers
                    for event in events_in_segment:
                        # Peak marker (red triangle)
                        peak_time = event['t_peak']
                        if sa <= peak_time <= sb:
                            peak_idx = int(round((peak_time - sa) * fs))
                            if 0 <= peak_idx < len(t_local):
                                ax.scatter(peak_time, data['dFF'][roi, slc][peak_idx], 
                                         c='red', s=25, marker='^', zorder=10, 
                                         edgecolors='darkred', linewidth=1)
                        
                        # Onset marker (green circle)
                        onset_time = event['t_on']
                        if sa <= onset_time <= sb:
                            onset_idx = int(round((onset_time - sa) * fs))
                            if 0 <= onset_idx < len(t_local):
                                ax.scatter(onset_time, data['dFF'][roi, slc][onset_idx], 
                                         c='green', s=15, marker='o', zorder=9,
                                         edgecolors='darkgreen', linewidth=1)
                        
                        # Offset marker (blue square)
                        offset_time = event['t_off']
                        if sa <= offset_time <= sb:
                            offset_idx = int(round((offset_time - sa) * fs))
                            if 0 <= offset_idx < len(t_local):
                                ax.scatter(offset_time, data['dFF'][roi, slc][offset_idx], 
                                         c='blue', s=15, marker='s', zorder=9,
                                         edgecolors='darkblue', linewidth=1)
                    
                    # Enhanced title with event statistics
                    if n_events > 0:
                        total_amp = sum(e['amp'] for e in events_in_segment)
                        mean_rise = np.mean([e['rise_s'] for e in events_in_segment])
                        mean_decay = np.mean([e['decay_s'] for e in events_in_segment])
                        ax.set_title(f'dF/F (raw) - {n_events} Ca events, Σamp={total_amp:.2f}, rise/decay={mean_rise:.2f}s/{mean_decay:.2f}s')
                    else:
                        ax.set_title('dF/F (raw) - Ca events detected')
                else:
                    ax.set_title('dF/F (raw) - Ca events detected')
            else:
                ax.set_title('dF/F (raw) - no Ca events')
        else:
            ax.set_title('dF/F (raw)')
        
        ax.set_ylabel('dFF')
        ax.set_xlim(sa, sb)
        # Add horizontal line at 0 if within range
        ylim = ax.get_ylim()
        if ylim[0] <= 0 <= ylim[1]:
            ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        
        # 5. dFF clean with enhanced Ca²⁺ transient visualization
        ax = axes[r]; r += 1
        ax.plot(t_local, data['dFF_clean'][roi, slc], lw=0.5, color='purple')
        ax.plot(t_local, data['dFF_smoothed'][roi, slc], lw=0.5, color='red')
        
        # Add enhanced Ca transient shading for clean trace
        if 'ca_masks' in qc and len(qc['ca_masks']) > roi:
            ca_mask_full = qc['ca_masks'][roi]
            ca_mask_segment = ca_mask_full[slc]
            
            if np.any(ca_mask_segment):
                baseline = np.min(data['dFF_clean'][roi, slc])
                ax.fill_between(t_local, baseline, data['dFF_clean'][roi, slc], 
                               where=ca_mask_segment, alpha=0.3, color='orange', 
                               label='Ca events', interpolate=True)
                
                # Enhanced event statistics for clean trace
                if 'ca_events' in qc and len(qc['ca_events']) > roi:
                    events_in_segment = [e for e in qc['ca_events'][roi] 
                                       if sa <= e['t_peak'] <= sb]
                    
                    if len(events_in_segment) > 0:
                        total_amp = sum(e['amp'] for e in events_in_segment)
                        total_area = sum(e['area'] for e in events_in_segment)
                        mean_snr = np.mean([e.get('snr', 0) for e in events_in_segment])
                        
                        # Mark peaks on clean trace
                        for event in events_in_segment:
                            peak_time = event['t_peak']
                            if sa <= peak_time <= sb:
                                peak_idx = int(round((peak_time - sa) * fs))
                                if 0 <= peak_idx < len(t_local):
                                    ax.scatter(peak_time, data['dFF_clean'][roi, slc][peak_idx], 
                                             c='red', s=20, marker='^', zorder=10,
                                             edgecolors='darkred', linewidth=1)
                        
                        ax.set_title(f'dF/F (clean) - {len(events_in_segment)} events, Σamp={total_amp:.2f}, area={total_area:.2f}')
                    else:
                        ax.set_title('dF/F (clean) - Ca events detected')
                else:
                    ax.set_title('dF/F (clean) - Ca events detected')
            else:
                ax.set_title('dF/F (clean) - no Ca events')
        else:
            ax.set_title('dF/F (clean)')
        
        ax.set_ylabel('dFF*')
        ax.set_xlim(sa, sb)
        # Add horizontal line at 0 if within range
        ylim = ax.get_ylim()
        if ylim[0] <= 0 <= ylim[1]:
            ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    
    if save:
        out_dir = rv['save_dir']
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"roi_{roi:04d}.png")
        fig.savefig(path, dpi=dpi)
        print(f"Enhanced review figure saved: {path}")
    else:
        print(f"Enhanced review figure displayed")
    
    return fig


def batch_plot_rois(data: Dict[str, Any], cfg: Dict[str, Any], roi_indices, limit: Optional[int] = None) -> None:
    """Batch export review figures"""
    print(f"\n=== BATCH PLOTTING ROIS ===")
    
    if limit is not None:
        roi_indices = roi_indices[:limit]
    
    print(f"Plotting {len(roi_indices)} ROIs...")
    
    rv = cfg['review']
    save_dir = rv['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    for i, rid in enumerate(roi_indices):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(roi_indices)}")
        
        fig = plot_roi_review(data, cfg, rid, save=True)
        plt.close(fig)
    
    print(f"Batch plotting complete: {len(roi_indices)} figures saved to {save_dir}")

# Replace the entire enhanced feature extraction and classification section with this scale-aware version:

def compute_orientation_coherence(A: np.ndarray, ypix: np.ndarray, xpix: np.ndarray, H: int, W: int) -> float:
    """Compute orientation coherence from structure tensor (legacy version for compatibility)"""
    return compute_scale_aware_coherence(A, ypix, xpix, H, W, window_px=3)


# Update the extract_scale_aware_roi_features function to include principal angle calculation:




def extract_scale_aware_roi_features(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """Extract scale-aware ROI features using µm measurements where appropriate"""
    print("\n=== EXTRACTING SCALE-AWARE ROI FEATURES ===")
    
    stat_list = data.get('stat')
    if stat_list is None:
        raise ValueError("stat list missing from data")
    
    # Get imaging metadata
    metadata = data.get('imaging_metadata')
    if metadata is None:
        raise ValueError("imaging_metadata missing - run load_imaging_metadata first")
    
    pixel_size_um = metadata.get('pixel_size_um', metadata.get('microns_per_pixel_x', 0.5))
    print(f"Using pixel size: {pixel_size_um:.6f} µm/pixel")
    
    n_rois = data['F'].shape[0]
    print(f"Processing {n_rois} ROIs with scale-aware features")
    
    # Get anatomy image
    A = data.get('anatomy_image')
    if A is None:
        print("  Anatomy image not cached, selecting...")
        A = select_anatomical_image(data, cfg)
    else:
        print(f"  Using cached anatomy image: {A.shape}")
    
    H, W = A.shape
    
    # Scale-aware parameters
    core_percentile = 70.0  # Use top 70% of lambda values for core
    coherence_window_um = 2.0  # µm window for structure tensor
    coherence_window_px = max(3, int(round(coherence_window_um / pixel_size_um)))
    morph_smooth_um = 0.3  # µm smoothing before skeletonization
    morph_smooth_px = max(1, int(round(morph_smooth_um / pixel_size_um)))
    
    print(f"Scale-aware parameters:")
    print(f"  Core percentile: {core_percentile}% (scale-invariant)")
    print(f"  Coherence window: {coherence_window_um} µm = {coherence_window_px} px")
    print(f"  Morphological smoothing: {morph_smooth_um} µm = {morph_smooth_px} px")
    
    # Initialize feature arrays
    features = {
        # Basic properties
        'area_px': np.zeros(n_rois, dtype=np.int32),
        'area_um2': np.zeros(n_rois, dtype=np.float32),
        'area_core_px': np.zeros(n_rois, dtype=np.int32),
        'centroid_x': np.zeros(n_rois, dtype=np.float32),
        'centroid_y': np.zeros(n_rois, dtype=np.float32),
        
        # Scale-invariant shape features
        'aspect_ratio': np.zeros(n_rois, dtype=np.float32),      # major/minor (invariant)
        'solidity': np.zeros(n_rois, dtype=np.float32),          # area/convex_hull (invariant)
        'circularity': np.zeros(n_rois, dtype=np.float32),       # 4π*area/perimeter² (invariant)
        'eccentricity': np.zeros(n_rois, dtype=np.float32),      # sqrt(1-minor²/major²) (invariant)
        'principal_angle_deg': np.zeros(n_rois, dtype=np.float32),  # NEW: angle of major axis for orientation deltas
        
        # Scale-dependent features (in µm)
        'thickness_um': np.zeros(n_rois, dtype=np.float32),      # 2x median distance transform
        'skeleton_length_um': np.zeros(n_rois, dtype=np.float32), # skeleton path length
        'major_axis_um': np.zeros(n_rois, dtype=np.float32),     # major axis length
        'minor_axis_um': np.zeros(n_rois, dtype=np.float32),     # minor axis length
        
        # Process-specific features
        'orientation_coherence': np.zeros(n_rois, dtype=np.float32), # (µm-window based)
        'branch_count': np.zeros(n_rois, dtype=np.int32),            # skeletal branchpoints
        
        # Intensity features (scale-invariant)
        'intensity_mean': np.zeros(n_rois, dtype=np.float32),
        'intensity_contrast_z': np.zeros(n_rois, dtype=np.float32),
    }
    
    # Global anatomy statistics
    g_mean = float(np.nanmean(A))
    g_std = float(np.nanstd(A) + 1e-9)
    
    # Processing counters
    oob_count = 0
    zero_area_count = 0
    small_roi_count = 0
    core_extraction_count = 0
    
    print("  Processing ROIs with scale-aware feature extraction...")
    for i, st in enumerate(stat_list):
        if i % 50 == 0 and i > 0:
            print(f"    Processed {i}/{n_rois} ROIs...")
        
        ypix, xpix, lam = _roi_pixels_from_stat(st)
        original_size = ypix.size
        
        # Bounds clipping
        ypix, xpix, lam = _clip_roi_pixels_to_image(ypix, xpix, lam, H, W)
        clipped_size = ypix.size
        
        if clipped_size < original_size:
            oob_count += 1
        
        features['area_px'][i] = clipped_size
        features['area_um2'][i] = clipped_size * (pixel_size_um ** 2)
        
        if clipped_size == 0:
            zero_area_count += 1
            continue
        
        if clipped_size < 3:
            small_roi_count += 1
            continue
        
        # Extract core pixels for stable shape analysis
        ypix_core, xpix_core = compute_roi_core_mask(ypix, xpix, lam, core_percentile)
        features['area_core_px'][i] = len(ypix_core)
        
        if len(ypix_core) >= 3:
            core_extraction_count += 1
            shape_ypix, shape_xpix = ypix_core, xpix_core
        else:
            shape_ypix, shape_xpix = ypix, xpix
        
        # Centroid (lambda-weighted, all pixels)
        w = lam / (lam.sum() + 1e-9)
        features['centroid_y'][i] = (ypix * w).sum()
        features['centroid_x'][i] = (xpix * w).sum()
        
        # Scale-invariant shape features (PCA on core pixels) + principal angle
        if len(shape_ypix) >= 3:
            coords = np.vstack([shape_xpix, shape_ypix]).T.astype(np.float32)
            coords_um = coords * pixel_size_um
            c0 = coords_um - coords_um.mean(0)
            
            # PCA basis - use SVD for more stable computation
            _, _, Vt = np.linalg.svd(c0, full_matrices=False)
            u_major = Vt[0]  # Major axis unit vector
            u_minor = Vt[1]  # Minor axis unit vector
            
            # Project onto axes and measure extents (Feret diameters in µm)
            t_major = c0 @ u_major
            t_minor = c0 @ u_minor
            major_len_um = float(t_major.max() - t_major.min())
            minor_len_um = float(t_minor.max() - t_minor.min())
            
            # Store actual Feret diameters (not σ)
            features['major_axis_um'][i] = major_len_um
            features['minor_axis_um'][i] = minor_len_um
            features['aspect_ratio'][i] = major_len_um / (minor_len_um + 1e-6)
            features['eccentricity'][i] = np.sqrt(max(0.0, 1.0 - (minor_len_um/max(major_len_um, 1e-6))**2))
            
            # Principal angle of major axis in degrees [0, 180)
            ang = np.degrees(np.arctan2(u_major[1], u_major[0])) % 180.0
            features['principal_angle_deg'][i] = ang
            
            print(f"    ROI {i}: Feret major={major_len_um:.2f} µm, minor={minor_len_um:.2f} µm") if i < 5 else None
        else:
            features['major_axis_um'][i] = 0.0
            features['minor_axis_um'][i] = 0.0
            features['aspect_ratio'][i] = 1.0
            features['eccentricity'][i] = 0.0
            features['principal_angle_deg'][i] = 0.0
        
        # Solidity (scale-invariant)
        features['solidity'][i] = compute_solidity(shape_ypix, shape_xpix, H, W)
        
        # Circularity (scale-invariant)
        features['circularity'][i] = compute_circularity(shape_ypix, shape_xpix, H, W)
        
        # Scale-dependent thickness (µm)
        thickness_px = compute_thickness_px(ypix, xpix, H, W)
        features['thickness_um'][i] = thickness_px * pixel_size_um
        
        # Scale-dependent skeleton features (µm)
        skeleton_data = compute_scale_aware_skeleton(ypix, xpix, H, W, morph_smooth_px)
        features['skeleton_length_um'][i] = skeleton_data['length_px'] * pixel_size_um
        features['branch_count'][i] = skeleton_data['branch_count']
        
        # Scale-aware orientation coherence (µm window)
        features['orientation_coherence'][i] = compute_scale_aware_coherence(
            A, ypix, xpix, H, W, coherence_window_px
        )
        
        # Intensity features (scale-invariant)
        intensities = A[ypix, xpix]
        features['intensity_mean'][i] = np.nanmean(intensities)
        features['intensity_contrast_z'][i] = (features['intensity_mean'][i] - g_mean) / g_std
    
    data['roi_features'] = features
    
    # Summary statistics including new principal angle feature
    areas_um2 = features['area_um2']
    thicknesses_um = features['thickness_um']
    lengths_um = features['skeleton_length_um']
    aspect_ratios = features['aspect_ratio']
    angles_deg = features['principal_angle_deg']
    
    print(f"\nScale-aware feature extraction complete:")
    print(f"  Processing summary:")
    print(f"    Out-of-bounds: {oob_count}/{n_rois} ROIs")
    print(f"    Zero area: {zero_area_count} ROIs")
    print(f"    Small ROIs: {small_roi_count} ROIs")
    print(f"    Core extraction used: {core_extraction_count}/{n_rois} ROIs")
    
    valid_mask = areas_um2 > 0
    if valid_mask.sum() > 0:
        print(f"  Feature ranges (valid ROIs only):")
        print(f"    Area: [{areas_um2[valid_mask].min():.1f}, {areas_um2[valid_mask].max():.1f}] µm², median: {np.median(areas_um2[valid_mask]):.1f}")
        print(f"    Thickness: [{thicknesses_um[valid_mask].min():.2f}, {thicknesses_um[valid_mask].max():.2f}] µm, median: {np.median(thicknesses_um[valid_mask]):.2f}")
        print(f"    Skeleton length: [{lengths_um[valid_mask].min():.1f}, {lengths_um[valid_mask].max():.1f}] µm, median: {np.median(lengths_um[valid_mask]):.1f}")
        print(f"    Aspect ratio: [{aspect_ratios[valid_mask].min():.2f}, {aspect_ratios[valid_mask].max():.2f}], median: {np.median(aspect_ratios[valid_mask]):.2f}")
        print(f"    Principal angles: [{angles_deg[valid_mask].min():.1f}, {angles_deg[valid_mask].max():.1f}] degrees, median: {np.median(angles_deg[valid_mask]):.1f}")

# Add helper functions for tie-breaker logic
def neighbor_snap_to_process(data: Dict[str, Any], pixel_size_um: float, ang_map: np.ndarray, 
                           max_edge_dist_um: float = 4.0, max_dtheta_deg: float = 20.0) -> None:
    """
    Promote uncertain ROIs to process if they are close to existing process ROIs 
    and have similar orientation (neighbor-snap tie-breaker)
    """
    print(f"\n  Applying neighbor-snap tie-breaker:")
    print(f"    Max edge distance: {max_edge_dist_um} µm")
    print(f"    Max orientation difference: {max_dtheta_deg} degrees")
    print(f"    Using local field orientation from {ang_map.shape} angle map")
    
    H, W = data['anatomy_image'].shape
    feats = data['roi_features']
    labels = np.array(data['roi_labels'], dtype=object)
    angles = feats.get('principal_angle_deg', np.zeros(len(labels)))
    
    # Build union mask of current process ROIs
    proc_idx = np.where(labels == 'process')[0]
    if proc_idx.size == 0:
        print(f"    No process ROIs found - skipping neighbor-snap")
        data['roi_labels'] = labels.tolist()
        return
    
    print(f"    Building process union mask from {len(proc_idx)} process ROIs...")
    proc_mask = np.zeros((H, W), dtype=bool)
    
    for i in proc_idx:
        y, x, _ = _roi_pixels_from_stat(data['stat'][i])
        y, x, _ = _clip_roi_pixels_to_image(y, x, _, H, W)
        if len(y) > 0:
            proc_mask[y, x] = True
    
    if not proc_mask.any():
        print(f"    No valid process pixels found - skipping neighbor-snap")
        data['roi_labels'] = labels.tolist()
        return
    
    # Calculate distance transform to find proximity to process pixels
    from scipy.ndimage import distance_transform_edt
    dist_px = distance_transform_edt(~proc_mask)
    
    # Check uncertain ROIs for promotion criteria
    unc_idx = np.where(labels == 'uncertain')[0]
    promoted_count = 0
    skipped_weak_coherence = 0
    skipped_far = 0
    skipped_misaligned = 0
    skipped_oob = 0
    
    print(f"    Evaluating {len(unc_idx)} uncertain ROIs for neighbor-snap promotion...")
    
    for i in unc_idx:
        y, x, _ = _roi_pixels_from_stat(data['stat'][i])
        y, x, _ = _clip_roi_pixels_to_image(y, x, _, H, W)
        
        if len(y) == 0:
            continue
        
        # Check proximity: minimum distance to any process pixel
        d_um = float(np.min(dist_px[y, x])) * pixel_size_um
        
        if d_um > max_edge_dist_um:
            skipped_far += 1
            continue
        
        # Check orientation coherence as process quality gate
        coh = feats['orientation_coherence'][i]
        if coh < 0.50:  # Weak coherence indicates poor line-like structure
            skipped_weak_coherence += 1
            continue
        
        # Get local field angle at ROI centroid
        cx = int(round(feats['centroid_x'][i]))
        cy = int(round(feats['centroid_y'][i]))
        
        if cy < 0 or cy >= ang_map.shape[0] or cx < 0 or cx >= ang_map.shape[1]:
            skipped_oob += 1
            continue
        
        # Use local field angle instead of global median
        roi_ang = angles[i]
        ref_ang = float(ang_map[cy, cx])
        
        # Calculate angular difference, wrapped to [0, 90] degrees
        dtheta = abs(((roi_ang - ref_ang + 90) % 180) - 90)
        
        if dtheta <= max_dtheta_deg:
            labels[i] = 'process'
            promoted_count += 1
        else:
            skipped_misaligned += 1
    
    print(f"    Neighbor-snap results:")
    print(f"      Promoted to process: {promoted_count}")
    print(f"      Skipped (too far): {skipped_far}")
    print(f"      Skipped (weak coherence): {skipped_weak_coherence}")  
    print(f"      Skipped (misaligned): {skipped_misaligned}")
    print(f"      Skipped (out of bounds): {skipped_oob}")
    
    data['roi_labels'] = labels.tolist()

def _roi_touches_border(stat_entry: dict, H: int, W: int, margin_px: int = 4) -> bool:
    """Check if ROI touches image border within specified margin"""
    y, x, _ = _roi_pixels_from_stat(stat_entry)
    y, x, _ = _clip_roi_pixels_to_image(y, x, _, H, W)
    
    if len(y) == 0:
        return False
    
    return (y.min() < margin_px) or (x.min() < margin_px) or \
           (y.max() > H - 1 - margin_px) or (x.max() > W - 1 - margin_px)

def apply_soma_halo_promotion(data: Dict[str, Any], pixel_size_um: float, radius_um: float = 3.0, circularity_min: float = 0.70) -> None:
    """
    Promote high-circularity uncertain ROIs near soma to soma class (soma halo effect)
    """
    print(f"\n  Applying soma halo promotion:")
    print(f"    Halo radius: {radius_um} µm")
    print(f"    Minimum circularity: {circularity_min}")
    
    feats = data['roi_features']
    labels = np.array(data['roi_labels'], dtype=object)
    
    # Get soma centroids for spatial indexing (include Cellpose somas as seeds)
    soma_idx = np.where((labels == 'soma') | (labels == 'soma_cp'))[0]
    if soma_idx.size == 0:
        print(f"    No soma or soma_cp ROIs found - skipping halo promotion")
        return
    
    print(f"    Building spatial index from {len(soma_idx)} soma/soma_cp ROIs...")
    soma_xy = np.column_stack([feats['centroid_x'][soma_idx], feats['centroid_y'][soma_idx]])
    
    from scipy.spatial import cKDTree
    tree = cKDTree(soma_xy)
    
    # Check uncertain ROIs for halo promotion
    unc_idx = np.where(labels == 'uncertain')[0]
    promoted_count = 0
    skipped_far = 0
    skipped_low_circularity = 0
    
    print(f"    Evaluating {len(unc_idx)} uncertain ROIs for halo promotion...")
    
    for i in unc_idx:
        cx, cy = feats['centroid_x'][i], feats['centroid_y'][i]
        
        # Find distance to nearest soma centroid
        d_px, _ = tree.query([cx, cy], k=1)
        d_um = d_px * pixel_size_um  # Convert to microns
        
        if d_um > radius_um:
            skipped_far += 1
            continue
        
        # Check circularity requirement (blob-like shape near soma)
        circ = feats['circularity'][i]
        if circ < circularity_min:
            skipped_low_circularity += 1
            continue
        
        # Promote to soma
        labels[i] = 'soma'
        promoted_count += 1
    
    print(f"    Soma halo results:")
    print(f"      Promoted to soma: {promoted_count}")
    print(f"      Skipped (too far): {skipped_far}")
    print(f"      Skipped (low circularity): {skipped_low_circularity}")
    
    data['roi_labels'] = labels.tolist()

def apply_border_uncertainty_policy(data: Dict[str, Any], margin_px: int = 4) -> None:
    """
    Demote border-touching process ROIs to uncertain (border artifact guard)
    """
    print(f"\n  Applying border uncertainty policy:")
    print(f"    Border margin: {margin_px} pixels")
    
    H, W = data['anatomy_image'].shape
    labels = np.array(data['roi_labels'], dtype=object)
    
    # Check process ROIs for border contact
    proc_idx = np.where(labels == 'process')[0]
    demoted_count = 0
    
    print(f"    Checking {len(proc_idx)} process ROIs for border contact...")
    
    for i in proc_idx:
        if _roi_touches_border(data['stat'][i], H, W, margin_px):
            labels[i] = 'uncertain'
            demoted_count += 1
    
    print(f"    Border policy results:")
    print(f"      Demoted to uncertain: {demoted_count}")
    
    data['roi_labels'] = labels.tolist()

# Update the main classification function with new process rules and tie-breakers
def classify_rois_scale_aware(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """Scale-aware soma/process classification using µm thresholds with enhanced tie-breaker logic"""
    print("\n=== SCALE-AWARE ROI CLASSIFICATION WITH TIE-BREAKERS ===")
    
    feats = data.get('roi_features')
    metadata = data.get('imaging_metadata')
    
    if feats is None:
        raise ValueError("Features missing - run extract_scale_aware_roi_features first")
    if metadata is None:
        raise ValueError("Imaging metadata missing")
    
    pixel_size_um = metadata.get('pixel_size_um', 0.5)
    objective_na = metadata.get('objective_na')
    
    print(f"Scale-aware classification with pixel size: {pixel_size_um:.3f} µm/pixel")
    if objective_na:
        print(f"  Objective NA: {objective_na:.2f}")
    
    # UPDATED: Tightened physical size guards for soma
    print(f"\nEnhanced classification criteria:")
    
    # Cerebellum-friendly soma criteria (granule cells are ~35-70 µm²)
    soma_min_area_um2 = 50.0        # Lowered from 100 for granule cells
    soma_max_area_um2 = 400.0       # Upper limit for artifact rejection
    soma_min_diameter_um = 4.0      # Lowered from 5.0 for small somas
    soma_max_diameter_um = 25.0     # Maximum soma diameter
    soma_max_aspect_ratio = 2.2     # Scale-invariant shape constraint
    soma_min_solidity = 0.60        # Scale-invariant compactness
    soma_min_circularity = 0.50     # Scale-invariant roundness
    
    print(f"  SOMA (must meet ALL 6 criteria) - CEREBELLUM-FRIENDLY:")
    print(f"    area_um2: [{soma_min_area_um2}, {soma_max_area_um2}] µm² (lowered for granule cells)")
    print(f"    major_axis_um: [{soma_min_diameter_um}, {soma_max_diameter_um}] µm (Feret diameter)")
    print(f"    aspect_ratio <= {soma_max_aspect_ratio} (scale-invariant)")
    print(f"    solidity >= {soma_min_solidity} (scale-invariant)")
    print(f"    circularity >= {soma_min_circularity} (scale-invariant)")
    
    # NEW: Process criteria using ANY of three paired cues (µm-aware)
    # Remove global thickness constraint - each cue has its own requirements
    print(f"  PROCESS (meet ANY of 3 paired cues) - NEW LOGIC:")
    print(f"    Cue 1 - Thin & Coherent: thickness_um <= 2.4 AND coherence >= 0.55")
    print(f"    Cue 2 - Long Filament: skeleton_length_um >= 8.0")  
    print(f"    Cue 3 - Slender Shape: aspect_ratio >= 1.9 AND circularity <= 0.85")
    print(f"    (No global thickness requirement - each cue self-validates)")
    
    # Extract features
    area_um2 = feats['area_um2']
    major_axis_um = feats['major_axis_um']
    thickness_um = feats['thickness_um']
    skeleton_length_um = feats['skeleton_length_um']
    aspect_ratio = feats['aspect_ratio']
    solidity = feats['solidity']
    circularity = feats['circularity']
    orientation_coherence = feats['orientation_coherence']
    
    n_rois = len(area_um2)
    print(f"\nEvaluating {n_rois} ROIs with enhanced scale-aware criteria...")
    
    # Soma criteria evaluation (tightened)
    soma_area = (area_um2 >= soma_min_area_um2) & (area_um2 <= soma_max_area_um2)
    soma_diameter = (major_axis_um >= soma_min_diameter_um) & (major_axis_um <= soma_max_diameter_um)
    soma_aspect = aspect_ratio <= soma_max_aspect_ratio
    soma_solid = solidity >= soma_min_solidity
    soma_circ = circularity >= soma_min_circularity
    
    soma_all = soma_area & soma_diameter & soma_aspect & soma_solid & soma_circ
    
    # NEW: Process criteria - ANY of three paired cues
    thin_and_coherent = (thickness_um <= 2.4) & (orientation_coherence >= 0.55)
    long_filament = (skeleton_length_um >= 8.0)
    slender_shape = (aspect_ratio >= 1.9) & (circularity <= 0.85)
    
    # Process rule: meet ANY of the three cues (no global thickness constraint)
    proc_all = thin_and_coherent | long_filament | slender_shape
    
    print(f"  Enhanced criteria breakdown:")
    print(f"    Soma area [{soma_min_area_um2}, {soma_max_area_um2}] µm²: {soma_area.sum()}/{n_rois} ({100*soma_area.sum()/n_rois:.1f}%)")
    print(f"    Soma diameter [{soma_min_diameter_um}, {soma_max_diameter_um}] µm: {soma_diameter.sum()}/{n_rois} ({100*soma_diameter.sum()/n_rois:.1f}%)")
    print(f"    Soma aspect <= {soma_max_aspect_ratio}: {soma_aspect.sum()}/{n_rois} ({100*soma_aspect.sum()/n_rois:.1f}%)")
    print(f"    Soma solidity >= {soma_min_solidity}: {soma_solid.sum()}/{n_rois} ({100*soma_solid.sum()/n_rois:.1f}%)")
    print(f"    Soma circularity >= {soma_min_circularity}: {soma_circ.sum()}/{n_rois} ({100*soma_circ.sum()/n_rois:.1f}%)")
    print(f"    ALL soma criteria: {soma_all.sum()}/{n_rois} ({100*soma_all.sum()/n_rois:.1f}%)")
    print()
    print(f"    Process cue 1 (thin & coherent): {thin_and_coherent.sum()}/{n_rois} ({100*thin_and_coherent.sum()/n_rois:.1f}%)")
    print(f"    Process cue 2 (long filament): {long_filament.sum()}/{n_rois} ({100*long_filament.sum()/n_rois:.1f}%)")
    print(f"    Process cue 3 (slender shape): {slender_shape.sum()}/{n_rois} ({100*slender_shape.sum()/n_rois:.1f}%)")
    print(f"    ANY process cue: {proc_all.sum()}/{n_rois} ({100*proc_all.sum()/n_rois:.1f}%)")
    
    # Optional Cellpose refinement (applied BEFORE tie-breakers)
    iou_soma_force = np.zeros(n_rois, dtype=bool)
    iou_process_allow = np.ones(n_rois, dtype=bool)
    
    if 'cellpose_iou' in data:
        print("\n  Applying optional Cellpose IoU refinement...")
        iou = data['cellpose_iou']
        iou_hi = 0.35  # Force soma if high IoU
        iou_lo = 0.15  # Allow process if low IoU
        
        iou_soma_force = iou >= iou_hi
        iou_process_allow = iou <= iou_lo
        
        print(f"    IoU >= {iou_hi} (force soma): {iou_soma_force.sum()}/{n_rois} ({100*iou_soma_force.sum()/n_rois:.1f}%)")
        print(f"    IoU <= {iou_lo} (allow process): {iou_process_allow.sum()}/{n_rois} ({100*iou_process_allow.sum()/n_rois:.1f}%)")
        print(f"    IoU middle zone: {(~iou_soma_force & ~iou_process_allow).sum()}/{n_rois}")
    
    # Primary classification logic
    labels = []
    soma_count = 0
    process_count = 0
    uncertain_count = 0
    iou_forced_count = 0
    
    print("\n  Applying primary classification logic...")
    for i in range(n_rois):
        if iou_soma_force[i]:
            # Cellpose forces soma (optional refinement)
            labels.append('soma')
            soma_count += 1
            iou_forced_count += 1
        elif soma_all[i]:
            # Meets all enhanced soma criteria
            labels.append('soma')
            soma_count += 1
        elif proc_all[i] and iou_process_allow[i]:
            # Meets enhanced process criteria and Cellpose allows
            labels.append('process')
            process_count += 1
        else:
            # Gap zone - will be processed by tie-breakers
            labels.append('uncertain')
            uncertain_count += 1
    
    data['roi_labels'] = labels
    
    print(f"  Primary classification results:")
    print(f"    Soma: {soma_count}/{n_rois} ({100*soma_count/n_rois:.1f}%)")
    if iou_forced_count > 0:
        print(f"      (including {iou_forced_count} Cellpose-forced)")
    print(f"    Process: {process_count}/{n_rois} ({100*process_count/n_rois:.1f}%)")
    print(f"    Uncertain (pre-tie-breakers): {uncertain_count}/{n_rois} ({100*uncertain_count/n_rois:.1f}%)")
    
    # ENHANCED TIE-BREAKER SEQUENCE
    print(f"\n=== APPLYING TIE-BREAKER SEQUENCE ===")
    
    # Read tie-breaker parameters from config
    tb_cfg = cfg.get('labeling', {}).get('tie_breakers', {})
    neighbor_cfg = tb_cfg.get('neighbor_snap', {})
    field_cfg = tb_cfg.get('field_prior', {})
    halo_cfg = tb_cfg.get('soma_halo', {})
    speck_cfg = tb_cfg.get('soma_speck_guard', {})
    
    # Compute orientation field for true field-orientation prior
    print(f"\n  Computing global orientation field...")
    field_window = field_cfg.get('window_um', 2.0)
    ang_map, coh_map = compute_orientation_field(
        data['anatomy_image'], window_um=field_window, pixel_size_um=pixel_size_um
    )
    
    # 1. Neighbor-snap: promote fragments near process backbone (using local field angle)
    neighbor_snap_to_process(data, pixel_size_um, ang_map, 
                            max_edge_dist_um=neighbor_cfg.get('max_edge_dist_um', 4.0),
                            max_dtheta_deg=neighbor_cfg.get('max_dtheta_deg', 20.0))
    
    # 2. True field-orientation prior: promote well-aligned coherent structures
    print(f"\n  Applying true field-orientation prior:")
    labels = np.array(data['roi_labels'], dtype=object)
    unc_idx = np.where(labels == 'uncertain')[0]
    promoted_count = 0
    skipped_misaligned = 0
    skipped_weak_coherence = 0
    skipped_oob = 0
    
    # Use config parameters
    max_dtheta_field = field_cfg.get('max_dtheta_deg', 25.0)
    coherence_min_field = field_cfg.get('coherence_min', 0.60)
    
    print(f"    Evaluating {len(unc_idx)} uncertain ROIs against local field...")
    print(f"    Max angle difference: {max_dtheta_field}°")
    print(f"    Min coherence: {coherence_min_field}")
    
    for i in unc_idx:
        cx = int(round(feats['centroid_x'][i]))
        cy = int(round(feats['centroid_y'][i]))
        
        if cy < 0 or cy >= ang_map.shape[0] or cx < 0 or cx >= ang_map.shape[1]:
            skipped_oob += 1
            continue
        
        # Get local field properties
        local_ang = float(ang_map[cy, cx])
        local_coh = float(coh_map[cy, cx])
        
        # Get ROI properties
        roi_ang = float(feats['principal_angle_deg'][i])
        roi_coh = float(feats['orientation_coherence'][i])
        
        # Check alignment: folded |Δθ| into [0,90]
        dtheta = abs(((roi_ang - local_ang + 90) % 180) - 90)
        
        if dtheta > max_dtheta_field:
            skipped_misaligned += 1
            continue
        
        # Check coherence: either local field or ROI must be coherent
        max_coherence = max(local_coh, roi_coh)
        if max_coherence < coherence_min_field:
            skipped_weak_coherence += 1
            continue
        
        # Promote to process
        labels[i] = 'process'
        promoted_count += 1
    
    data['roi_labels'] = labels.tolist()
    
    print(f"    True field-orientation results:")
    print(f"      Promoted to process: {promoted_count}")
    print(f"      Skipped (misaligned > {max_dtheta_field}°): {skipped_misaligned}")
    print(f"      Skipped (weak coherence < {coherence_min_field}): {skipped_weak_coherence}")
    print(f"      Skipped (out of bounds): {skipped_oob}")
    
    # 3. Soma halo: promote high-circularity uncertain near soma
    halo_radius = halo_cfg.get('radius_um', 3.0)
    halo_circularity = halo_cfg.get('circularity_min', 0.70)
    apply_soma_halo_promotion(data, pixel_size_um, radius_um=halo_radius, circularity_min=halo_circularity)
    
    # 4. Soma speck guard: demote tiny isolated somas (AFTER halo rescue)
    speck_min_area = speck_cfg.get('min_area_um2', 100.0)
    speck_protect = speck_cfg.get('protect_labels', ['soma_cp'])
    apply_soma_speck_guard(data, min_area_um2=speck_min_area, protect_labels=speck_protect)
    
    # 5. Border uncertainty: demote border-touching process to uncertain
    border_margin = tb_cfg.get('border_uncertain_px', 4)
    apply_border_uncertainty_policy(data, margin_px=border_margin)
    
    # Final summary with tie-breaker impact
    final_labels = data['roi_labels']
    final_counts = Counter(final_labels)
    
    print(f"\n=== FINAL SCALE-AWARE CLASSIFICATION RESULTS ===")
    total = len(final_labels)
    final_soma = final_counts.get('soma', 0)
    final_process = final_counts.get('process', 0) 
    final_uncertain = final_counts.get('uncertain', 0)
    
    print(f"  Final classification:")
    print(f"    Soma: {final_soma}/{total} ({100*final_soma/total:.1f}%)")
    print(f"    Process: {final_process}/{total} ({100*final_process/total:.1f}%)")
    print(f"    Uncertain: {final_uncertain}/{total} ({100*final_uncertain/total:.1f}%)")
    
    print(f"  Tie-breaker impact:")
    print(f"    Process gain: +{final_process - process_count} ({100*(final_process - process_count)/total:.1f}%)")
    print(f"    Uncertain reduction: -{uncertain_count - final_uncertain} ({100*(uncertain_count - final_uncertain)/total:.1f}%)")
    
    # Physical size distributions for validation
    print(f"\nPhysical size validation:")
    soma_mask = np.array(final_labels) == 'soma'
    proc_mask = np.array(final_labels) == 'process'
    unc_mask = np.array(final_labels) == 'uncertain'
    
    if soma_mask.sum() > 0:
        soma_areas = area_um2[soma_mask]
        soma_diameters = major_axis_um[soma_mask]
        print(f"  Soma (n={soma_mask.sum()}):")
        print(f"    Areas: {soma_areas.mean():.1f} ± {soma_areas.std():.1f} µm² (range: {soma_areas.min():.1f}-{soma_areas.max():.1f})")
        print(f"    Diameters: {soma_diameters.mean():.1f} ± {soma_diameters.std():.1f} µm (range: {soma_diameters.min():.1f}-{soma_diameters.max():.1f})")
    
    if proc_mask.sum() > 0:
        proc_thickness = thickness_um[proc_mask]
        proc_lengths = skeleton_length_um[proc_mask]
        proc_coherence = orientation_coherence[proc_mask]
        print(f"  Process (n={proc_mask.sum()}):")
        print(f"    Thickness: {proc_thickness.mean():.2f} ± {proc_thickness.std():.2f} µm (range: {proc_thickness.min():.2f}-{proc_thickness.max():.2f})")
        print(f"    Coherence: {proc_coherence.mean():.3f} ± {proc_coherence.std():.3f} (range: {proc_coherence.min():.3f}-{proc_coherence.max():.3f})")
        proc_lengths_valid = proc_lengths[proc_lengths > 0]
        if len(proc_lengths_valid) > 0:
            print(f"    Lengths: {proc_lengths_valid.mean():.1f} ± {proc_lengths_valid.std():.1f} µm (range: {proc_lengths_valid.min():.1f}-{proc_lengths_valid.max():.1f})")
    
    if unc_mask.sum() > 0:
        unc_areas = area_um2[unc_mask]
        unc_thickness = thickness_um[unc_mask]
        unc_coherence = orientation_coherence[unc_mask]
        print(f"  Uncertain (n={unc_mask.sum()}):")
        print(f"    Areas: {unc_areas.mean():.1f} ± {unc_areas.std():.1f} µm²")
        print(f"    Thickness: {unc_thickness.mean():.2f} ± {unc_thickness.std():.2f} µm")
        print(f"    Coherence: {unc_coherence.mean():.3f} ± {unc_coherence.std():.3f}")
    
    print(f"\nScale-aware classification with tie-breakers complete")
    print(f"Resolution: {pixel_size_um:.3f} µm/pixel, Expected uncertain range: 10-15%")

def compute_scale_aware_skeleton(ypix: np.ndarray, xpix: np.ndarray, H: int, W: int, smooth_px: int) -> Dict[str, float]:
    """Compute skeleton features with µm-based preprocessing"""
    if len(ypix) < 5:
        return {'length_px': 0.0, 'branch_count': 0}
    
    # Create binary mask
    mask = np.zeros((H, W), dtype=bool)
    mask[ypix, xpix] = True
    
    # Apply morphological smoothing (µm-sized) to stabilize skeleton
    if smooth_px > 0:
        from scipy.ndimage import binary_opening, binary_closing
        struct = np.ones((2*smooth_px+1, 2*smooth_px+1))
        mask = binary_closing(binary_opening(mask, struct), struct)
    
    # Skeletonize
    try:
        skeleton = skeletonize(mask)
        skeleton_length = np.sum(skeleton)
        
        # Count branch points (pixels with >2 neighbors)
        if skeleton_length > 0:
            branch_count = count_skeleton_branches(skeleton)
        else:
            branch_count = 0
        
        return {'length_px': float(skeleton_length), 'branch_count': branch_count}
    except:
        return {'length_px': 0.0, 'branch_count': 0}

def count_skeleton_branches(skeleton: np.ndarray) -> int:
    """Count branch points in skeleton (pixels with >2 neighbors)"""
    from scipy.ndimage import convolve
    
    # 8-connectivity kernel
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1], 
                       [1, 1, 1]], dtype=np.int32)
    
    # Count neighbors for each skeleton pixel
    neighbor_count = convolve(skeleton.astype(np.int32), kernel, mode='constant')
    
    # Branch points have >2 neighbors
    branch_mask = skeleton & (neighbor_count > 2)
    return np.sum(branch_mask)

def compute_scale_aware_coherence(A: np.ndarray, ypix: np.ndarray, xpix: np.ndarray, H: int, W: int, window_px: int) -> float:
    """Compute orientation coherence with µm-sized window"""
    if len(ypix) < 5:
        return 0.0
    
    # Create dilated region (µm-sized window)
    mask = np.zeros((H, W), dtype=bool)
    mask[ypix, xpix] = True
    
    # Dilate by window size
    for _ in range(window_px):
        mask = binary_dilation(mask)
    
    # Extract local region
    ymin, ymax = max(0, ypix.min()-window_px), min(H, ypix.max()+window_px+1)
    xmin, xmax = max(0, xpix.min()-window_px), min(W, xpix.max()+window_px+1)
    
    region = A[ymin:ymax, xmin:xmax]
    region_mask = mask[ymin:ymax, xmin:xmax]
    
    if region.size == 0 or not region_mask.any():
        return 0.0
    
    # Compute gradients
    grad_y = sobel(region, axis=0)
    grad_x = sobel(region, axis=1)
    
    # Structure tensor components
    Ixx = grad_x * grad_x
    Iyy = grad_y * grad_y
    Ixy = grad_x * grad_y
    
    # Average over the dilated region
    mask_indices = region_mask.nonzero()
    if len(mask_indices[0]) == 0:
        return 0.0
    
    Sxx = np.mean(Ixx[mask_indices])
    Syy = np.mean(Iyy[mask_indices])
    Sxy = np.mean(Ixy[mask_indices])
    
    # Eigenvalues of structure tensor
    trace = Sxx + Syy
    det = Sxx * Syy - Sxy * Sxy
    
    if trace <= 1e-9:
        return 0.0
    
    discriminant = trace * trace - 4 * det
    if discriminant < 0:
        return 0.0
    
    lambda1 = 0.5 * (trace + np.sqrt(discriminant))
    lambda2 = 0.5 * (trace - np.sqrt(discriminant))
    
    # Coherence measure
    if lambda1 + lambda2 <= 1e-9:
        return 0.0
    
    coherence = (lambda1 - lambda2) / (lambda1 + lambda2)
    return max(0.0, min(1.0, coherence))

def compute_circularity(ypix: np.ndarray, xpix: np.ndarray, H: int, W: int) -> float:
    """Compute circularity = 4π*area/perimeter² (scale-invariant)"""
    if len(ypix) < 3:
        return 0.0
    
    # Create mask for perimeter estimation
    mask = np.zeros((H, W), dtype=bool)
    mask[ypix, xpix] = True
    
    # Estimate perimeter from boundary pixels
    eroded = binary_erosion(mask, structure=np.ones((3,3)))
    boundary = mask & (~eroded)
    perimeter = np.sum(boundary)
    
    if perimeter == 0:
        return 0.0
    
    area = len(ypix)
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    return min(1.0, circularity)

# Add this utility function near the other helper functions:

def compute_orientation_field(A: np.ndarray, window_um: float, pixel_size_um: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return angle(deg in [0,180)) and coherence maps using a Gaussian window."""
    from scipy.ndimage import gaussian_filter, sobel
    
    print(f"    Computing orientation field with {window_um} µm window...")
    H, W = A.shape
    
    # Structure tensor components
    Ix = sobel(A, axis=1)
    Iy = sobel(A, axis=0)
    Jxx = Ix * Ix
    Jyy = Iy * Iy  
    Jxy = Ix * Iy
    
    # Gaussian smoothing with µm-based sigma
    sigma_px = max(1.0, window_um / pixel_size_um)
    print(f"      Gaussian sigma: {sigma_px:.1f} pixels")
    
    Sxx = gaussian_filter(Jxx, sigma=sigma_px)
    Syy = gaussian_filter(Jyy, sigma=sigma_px)
    Sxy = gaussian_filter(Jxy, sigma=sigma_px)
    
    # Eigenvalue decomposition
    trace = Sxx + Syy
    diff = Sxx - Syy
    disc = np.clip(diff*diff + 4*Sxy*Sxy, 0, None)**0.5
    lam1 = 0.5 * (trace + disc)
    lam2 = 0.5 * (trace - disc)
    
    # Coherence measure
    coherence = np.where(trace > 1e-9, (lam1 - lam2) / (lam1 + lam2), 0.0)
    
    # Dominant orientation (0..180): 0.5*atan2(2Sxy, Sxx-Syy)
    ang = (0.5 * np.degrees(np.arctan2(2*Sxy, diff))) % 180.0
    
    print(f"      Field angle range: [{ang.min():.1f}, {ang.max():.1f}] degrees")
    print(f"      Field coherence range: [{coherence.min():.3f}, {coherence.max():.3f}]")
    
    return ang.astype(np.float32), np.clip(coherence, 0, 1).astype(np.float32)


# Add this function with other tie-breaker helpers:

def apply_soma_speck_guard(data: Dict[str, Any], min_area_um2: float = 100.0, protect_labels=('soma_cp',)) -> None:
    """
    Demote tiny isolated somas to uncertain unless they're halo-rescued or Cellpose-forced
    """
    print(f"\n  Applying soma speck guard:")
    print(f"    Minimum area: {min_area_um2} µm²")
    print(f"    Protected labels: {protect_labels}")
    
    feats = data['roi_features']
    labels = np.array(data['roi_labels'], dtype=object)
    
    # Identify protected ROIs (Cellpose-forced, etc.)
    protect_mask = np.isin(labels, protect_labels)
    
    # Find tiny somas that aren't protected
    tiny_soma_mask = (labels == 'soma') & (feats['area_um2'] < min_area_um2) & (~protect_mask)
    demoted_count = tiny_soma_mask.sum()
    
    # Demote to uncertain
    labels[tiny_soma_mask] = 'uncertain'
    data['roi_labels'] = labels.tolist()
    
    print(f"    Soma speck guard results:")
    print(f"      Demoted to uncertain: {demoted_count}")
    print(f"      Protected from demotion: {protect_mask.sum()}")

def export_roi_spatial_data(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """Export ROI spatial data for external analysis"""
    print(f"\n=== EXPORTING ROI SPATIAL DATA ===")
    
    # Create spatial DataFrame
    df = create_spatial_roi_dataframe(data)
    if df is None:
        print("Cannot create DataFrame without pandas")
        return
    
    
    folder = cfg["paths"]["qc_dir"]
    save_path = os.path.join(folder, 'roi_analysis_data.pkl')

    # Save as CSV
    csv_path = save_path.replace('.pkl', '.csv') if save_path.endswith('.pkl') else f"{save_path}_roi_spatial.csv"
    df.to_csv(csv_path, index=False)
    print(f"Spatial data exported to: {csv_path}")
    
    # # Save full data as pickle for complete reconstruction
    # import pickle
    # pkl_path = save_path if save_path.endswith('.pkl') else f"{save_path}_complete.pkl"
    
    # # Prepare data for export (remove large arrays if requested)
    # export_data = data.copy()
    
    # with open(pkl_path, 'wb') as f:
    #     pickle.dump(export_data, f)
    # print(f"Complete data exported to: {pkl_path}")
    
    # Print summary for verification
    print(f"\nExport summary:")
    print(f"  Total ROIs: {len(df)}")
    print(f"  Spatial features: {len(df.columns)}")
    print(f"  Index mapping: {data['roi_index_map']['n_filtered']} -> {data['roi_index_map']['n_original']}")









# Add visualization function for spatial distribution
def plot_spatial_roi_distribution(data: Dict[str, Any], cfg: Dict[str, Any], 
                                 color_by: str = 'roi_label', save: bool = True) -> Any:
    """Plot spatial distribution of ROIs colored by specified feature"""
    print(f"\n=== PLOTTING SPATIAL ROI DISTRIBUTION ===")
    print(f"Coloring by: {color_by}")
    
    # Create spatial DataFrame
    df = create_spatial_roi_dataframe(data)
    if df is None:
        print("Cannot create plot without pandas")
        return None
    
    # Get anatomy image for background
    A = data.get('anatomy_image')
    if A is None:
        A = select_anatomical_image(data, cfg)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Show anatomy background
    ov_cfg = cfg['overlay']
    lo, hi = np.percentile(A, (ov_cfg['bg_pmin'], ov_cfg['bg_pmax']))
    bg = np.clip((A - lo) / (hi - lo + 1e-9), 0, 1)
    ax.imshow(bg, cmap='gray', alpha=0.6)
    
    # Color mapping
    if color_by == 'roi_label':
        color_map = {'soma': 'red', 'soma_cp': 'darkred', 'process': 'cyan', 'uncertain': 'yellow'}
        colors = [color_map.get(label, 'gray') for label in df[color_by]]
        title = "ROI Spatial Distribution by Classification"
    elif color_by == 'area_um2' and 'area_um2' in df.columns:
        colors = df[color_by]
        title = "ROI Spatial Distribution by Area (µm²)"
    elif color_by == 'intensity_mean':
        colors = df[color_by]
        title = "ROI Spatial Distribution by Mean Intensity"
    else:
        colors = 'blue'
        title = f"ROI Spatial Distribution by {color_by}"
    
    # Scatter plot
    scatter = ax.scatter(df['centroid_x_px'], df['centroid_y_px'], 
                        c=colors, s=15, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add colorbar for continuous variables
    if color_by not in ['roi_label'] and hasattr(colors, '__len__') and len(set(colors)) > 5:
        plt.colorbar(scatter, ax=ax, label=color_by)
    
    # Add legend for categorical variables
    elif color_by == 'roi_label':
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[label], label=f"{label} (n={sum(df[color_by] == label)})")
                          for label in color_map.keys() if label in df[color_by].values]
        ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title(title)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.invert_yaxis()  # Match image coordinates
    
    # Add ROI count and physical scale if available
    metadata = data.get('imaging_metadata', {})
    pixel_size = metadata.get('pixel_size_um')
    info_text = f"n = {len(df)} ROIs"
    if pixel_size:
        info_text += f"\nPixel size: {pixel_size:.3f} µm"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save:
        save_dir = cfg['overlay']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'spatial_distribution_{color_by}.png')
        fig.savefig(save_path, dpi=cfg['overlay']['dpi'], bbox_inches='tight')
        print(f"Spatial distribution plot saved: {save_path}")
    
    plt.show()
    return fig

def create_roi_index_mapping(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create comprehensive ROI index mapping for spatial analysis"""
    print("\n=== CREATING ROI INDEX MAPPING ===")
    
    # Get current filtered indices
    n_filtered = data['F'].shape[0]
    filtered_indices = np.arange(n_filtered)
    
    # Get original Suite2p indices
    if 'roi_mask' in data:
        original_indices = np.where(data['roi_mask'])[0]
        print(f"Filtered ROIs: {n_filtered}, Original Suite2p indices: {len(original_indices)}")
    else:
        # No filtering was applied
        original_indices = np.arange(n_filtered)
        print(f"No filtering applied, using direct mapping: {n_filtered} ROIs")
    
    # Create bidirectional mapping
    index_map = {
        'filtered_to_original': dict(zip(filtered_indices, original_indices)),
        'original_to_filtered': dict(zip(original_indices, filtered_indices)),
        'n_filtered': n_filtered,
        'n_original': len(original_indices) if 'roi_mask' in data else n_filtered,
        'filtering_applied': 'roi_mask' in data,
        'original_indices': original_indices,
        'filtered_indices': filtered_indices
    }
    
    # Add to data
    data['roi_index_map'] = index_map
    
    print(f"Index mapping created:")
    print(f"  Filtered -> Original: {n_filtered} mappings")
    print(f"  Example: filtered ROI 0 -> original Suite2p ROI {original_indices[0]}")
    print(f"  Example: filtered ROI 5 -> original Suite2p ROI {original_indices[5] if len(original_indices) > 5 else 'N/A'}")
    
    return data

def get_original_roi_index(data: Dict[str, Any], filtered_idx: int) -> int:
    """Convert filtered ROI index to original Suite2p index"""
    if 'roi_index_map' not in data:
        create_roi_index_mapping(data)
    
    return data['roi_index_map']['filtered_to_original'][filtered_idx]

def get_filtered_roi_index(data: Dict[str, Any], original_idx: int) -> Optional[int]:
    """Convert original Suite2p index to filtered ROI index (None if filtered out)"""
    if 'roi_index_map' not in data:
        create_roi_index_mapping(data)
    
    return data['roi_index_map']['original_to_filtered'].get(original_idx)

def create_spatial_roi_dataframe(data: Dict[str, Any]) -> Any:
    """Create comprehensive DataFrame for spatial analysis and visualization"""
    print("\n=== CREATING SPATIAL ROI DATAFRAME ===")
    
    try:
        import pandas as pd
    except ImportError:
        print("pandas not available - skipping DataFrame creation")
        return None
    
    # Ensure index mapping exists
    if 'roi_index_map' not in data:
        create_roi_index_mapping(data)
    
    n_rois = data['F'].shape[0]
    feats = data['roi_features']
    labels = data.get('roi_labels', ['unknown'] * n_rois)
    metadata = data.get('imaging_metadata', {})
    
    # Build comprehensive DataFrame
    df_data = {
        # Index tracking
        'filtered_idx': np.arange(n_rois),
        'original_suite2p_idx': [get_original_roi_index(data, i) for i in range(n_rois)],
        
        # Spatial coordinates
        'centroid_x_px': feats['centroid_x'],
        'centroid_y_px': feats['centroid_y'],
        
        # Classification
        'roi_label': labels,
        
        # Basic features
        'area_px': feats.get('area_px', feats.get('area', np.zeros(n_rois))),
        'aspect_ratio': feats['aspect_ratio'],
        'circularity': feats.get('circularity', np.zeros(n_rois)),
        'solidity': feats.get('solidity', np.zeros(n_rois)),
        
        # Intensity features
        'intensity_mean': feats.get('intensity_mean', feats.get('inten_mean', np.zeros(n_rois))),
        'intensity_contrast_z': feats.get('intensity_contrast_z', feats.get('inten_contrast_z', np.zeros(n_rois))),
    }
    
    # Add physical size features if available
    pixel_size_um = metadata.get('pixel_size_um', metadata.get('microns_per_pixel_x'))
    if pixel_size_um is not None:
        print(f"Adding physical size features (pixel size: {pixel_size_um:.3f} µm/pixel)")
        df_data.update({
            'centroid_x_um': feats['centroid_x'] * pixel_size_um,
            'centroid_y_um': feats['centroid_y'] * pixel_size_um,
            'area_um2': feats.get('area_um2', feats.get('area_px', feats.get('area', np.zeros(n_rois))) * pixel_size_um**2),
            'major_axis_um': feats.get('major_axis_um', np.zeros(n_rois)),
            'minor_axis_um': feats.get('minor_axis_um', np.zeros(n_rois)),
            'thickness_um': feats.get('thickness_um', np.zeros(n_rois)),
            'skeleton_length_um': feats.get('skeleton_length_um', np.zeros(n_rois)),
        })
    
    # Add process-specific features if available
    if 'orientation_coherence' in feats:
        df_data['orientation_coherence'] = feats['orientation_coherence']
    if 'principal_angle_deg' in feats:
        df_data['principal_angle_deg'] = feats['principal_angle_deg']
    if 'branch_count' in feats:
        df_data['branch_count'] = feats['branch_count']
    
    # Add Cellpose IoU if available
    if 'cellpose_iou' in data:
        df_data['cellpose_iou'] = data['cellpose_iou']
    
    # Add neuropil correction coefficient if available
    if 'neuropil_a' in data:
        df_data['neuropil_alpha'] = data['neuropil_a']
    
    # Create DataFrame
    df = pd.DataFrame(df_data)
    
    # Add some computed columns
    df['is_soma'] = df['roi_label'].isin(['soma', 'soma_cp'])
    df['is_process'] = df['roi_label'] == 'process'
    df['is_uncertain'] = df['roi_label'] == 'uncertain'
    
    print(f"Spatial DataFrame created: {len(df)} ROIs x {len(df.columns)} features")
    print(f"Label distribution:")
    print(df['roi_label'].value_counts().to_string())
    
    return df





# def compute_qc_metrics(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
#     """STEP 4C: Compute QC metrics including spike rates, SNR, and Ca transients with enhanced detection"""
#     print("\n=== COMPUTING QC METRICS ===")
    
#     # Check for spike data
#     spike_version = None
#     if 'spks_oasis' in data:
#         spike_version = 'spks_oasis'
#     elif 'spks_oasis_recon' in data:
#         spike_version = 'spks_oasis_recon'
    
#     if spike_version is None:
#         print("No spike data available - skipping spike-based QC")
        
#     fs = float(cfg["acq"]["fs"])
    
#     # Get dF/F data for Ca transient detection
#     dff_clean = data.get('dFF_clean')
#     if dff_clean is None:
#         print("No dF/F data available - skipping Ca transient detection")
#         return
    
#     N, T = dff_clean.shape
#     print(f"Computing QC metrics for {N} ROIs x {T} frames at {fs:.1f} Hz")
    
#     # Enhanced Ca transient detection parameters
#     ca_cfg = cfg.get("ca_transients", {})
#     smooth_sigma_s = ca_cfg.get("smooth_sigma_s", 0.10)
#     k_on = ca_cfg.get("k_on", 2.5)
#     k_off = ca_cfg.get("k_off", 1.0)
#     rise_max_s = ca_cfg.get("rise_max_s", 0.30)
#     peak_win_s = ca_cfg.get("peak_win_s", 1.00)
#     decay_max_s = ca_cfg.get("decay_max_s", 2.50)
#     min_amp = ca_cfg.get("min_amp", 0.20)
    
#     # NEW: Enhanced biological constraints
#     max_decay_slope_per_s = ca_cfg.get("max_decay_slope_per_s", 2.0)
#     min_series_peak_dff = ca_cfg.get("min_series_peak_dff", 0.50)
#     series_timeout_s = ca_cfg.get("series_timeout_s", 3.0)
#     merge_gap_s = ca_cfg.get("merge_gap_s", 0.5)
#     min_snr = ca_cfg.get("min_snr", 2.0)
    
#     print(f"Enhanced Ca transient detection parameters:")
#     print(f"  Basic: smoothing={smooth_sigma_s}s, thresholds={k_on}σ/{k_off}σ, min_amp={min_amp}")
#     print(f"  Kinetics: rise_max={rise_max_s}s, decay_max={decay_max_s}s")
#     print(f"  NEW - Biological constraints:")
#     print(f"    Max decay slope: {max_decay_slope_per_s} ΔF/F/s")
#     print(f"    Min series peak: {min_series_peak_dff} ΔF/F")
#     print(f"    Series timeout: {series_timeout_s}s")
#     print(f"    Event merge gap: {merge_gap_s}s")
#     print(f"    Min SNR: {min_snr}")
    
#     # Detect Ca transients for each ROI with enhanced algorithm
#     ca_events_list = []
#     ca_masks_list = []
#     ca_event_counts = np.zeros(N, dtype=np.int32)
#     ca_event_rates = np.zeros(N, dtype=np.float32)
#     ca_mean_amplitudes = np.zeros(N, dtype=np.float32)
#     ca_mean_rise_times = np.zeros(N, dtype=np.float32)
#     ca_mean_decay_times = np.zeros(N, dtype=np.float32)
#     ca_total_active_time = np.zeros(N, dtype=np.float32)
    
#     print("  Detecting Ca transients with enhanced algorithm...")
#     for i in range(30):
#         if i % 50 == 0 and i > 0:
#             print(f"    Processed {i}/{N} ROIs...")
        
#         try:
#             events, mask = detect_ca_transients(
#                 dff_clean[i], fs,
#                 smooth_sigma_s=smooth_sigma_s,
#                 k_on=k_on, k_off=k_off,
#                 rise_max_s=rise_max_s,
#                 peak_win_s=peak_win_s,
#                 decay_max_s=decay_max_s,
#                 min_amp=min_amp,
#                 use_z=False,
#                 # Enhanced parameters
#                 max_decay_slope_per_s=max_decay_slope_per_s,
#                 min_series_peak_dff=min_series_peak_dff,
#                 series_timeout_s=series_timeout_s,
#                 merge_gap_s=merge_gap_s,
#                 min_snr=min_snr
#             )
            
#             ca_events_list.append(events)
#             ca_masks_list.append(mask)
            
#             # Summary statistics
#             ca_event_counts[i] = len(events)
#             ca_event_rates[i] = len(events) / (T / fs)  # events per second
#             ca_total_active_time[i] = np.sum(mask) / fs  # seconds of active time
            
#             if len(events) > 0:
#                 ca_mean_amplitudes[i] = np.mean([e['amp'] for e in events])
#                 ca_mean_rise_times[i] = np.mean([e['rise_s'] for e in events])
#                 ca_mean_decay_times[i] = np.mean([e['decay_s'] for e in events])
#             else:
#                 ca_mean_amplitudes[i] = 0.0
#                 ca_mean_rise_times[i] = 0.0
#                 ca_mean_decay_times[i] = 0.0
                
#         except Exception as e:
#             print(f"    Warning: Enhanced Ca transient detection failed for ROI {i}: {e}")
#             ca_events_list.append([])
#             ca_masks_list.append(np.zeros(T, dtype=bool))
    
#     # Process spike data if available
#     if spike_version is not None:
#         spikes = data[spike_version]
        
#         # Parameters for spike detection and binning
#         qc_cfg = cfg.get("qc_metrics", {})
#         k_mad = qc_cfg.get("onset_threshold_mad", 3.0)
#         bin_duration_s = qc_cfg.get("spike_rate_bin_s", 0.2)
        
#         print(f"  Computing spike rates from {spike_version}")
#         print(f"    Onset detection threshold: {k_mad} MAD")
#         print(f"    Spike rate binning: {bin_duration_s}s")
        
#         # Spike detection and rates
#         spike_onsets = spike_onset_mask(spikes, k_mad=k_mad)
#         spike_rate_hz_binned = spike_rate_hz(spikes, fs, bin_duration_s, k_mad=k_mad)
#         spike_rate_hz_instantaneous = spike_onsets.astype(np.float32) * fs
        
#         # Smoothed spike rates for plotting
#         from scipy.ndimage import gaussian_filter1d
#         smooth_sigma_frames = max(1.0, 0.1 * fs)
#         spike_rate_smoothed_hz = np.array([gaussian_filter1d(spike_rate_hz_instantaneous[i], 
#                                                              sigma=smooth_sigma_frames) 
#                                           for i in range(N)])
        
#         # Spike statistics
#         spike_metrics = {
#             'spike_raw': spikes,
#             'spike_onsets': spike_onsets,
#             'spike_rate_hz_instantaneous': spike_rate_hz_instantaneous,
#             'spike_rate_hz_smoothed': spike_rate_smoothed_hz,
#             'spike_count_onsets': np.sum(spike_onsets, axis=1),
#             'spike_rate_mean_hz': np.mean(spike_rate_hz_instantaneous, axis=1),
#             'spike_amplitude_max': np.max(spikes, axis=1),
#             'spike_cv_isi_onsets': compute_spike_cv_isi(spike_onsets, fs),
#         }
#     else:
#         spike_metrics = {}
    
#     # Combined QC metrics with enhanced Ca transient data
#     qc_metrics = {
#         # Enhanced Ca transient detection
#         'ca_events': ca_events_list,  # List of event dicts per ROI
#         'ca_masks': ca_masks_list,    # List of boolean masks per ROI
#         'ca_event_counts': ca_event_counts,
#         'ca_event_rates_hz': ca_event_rates,
#         'ca_mean_amplitudes': ca_mean_amplitudes,
#         'ca_mean_rise_times_s': ca_mean_rise_times,
#         'ca_mean_decay_times_s': ca_mean_decay_times,
#         'ca_total_active_time_s': ca_total_active_time,
#         'ca_activity_fraction': ca_total_active_time / (T / fs),  # fraction of time active
        
#         # Spike metrics (if available)
#         **spike_metrics,
        
#         # Signal quality
#         'snr_estimate': estimate_signal_noise_ratio(data),
#         'temporal_stability': compute_temporal_stability(data),
        
#         # Metadata with enhanced parameters
#         'ca_detection_params': {
#             'smooth_sigma_s': smooth_sigma_s,
#             'k_on': k_on, 'k_off': k_off,
#             'rise_max_s': rise_max_s, 'peak_win_s': peak_win_s, 'decay_max_s': decay_max_s,
#             'min_amp': min_amp,
#             # Enhanced constraints
#             'max_decay_slope_per_s': max_decay_slope_per_s,
#             'min_series_peak_dff': min_series_peak_dff,
#             'series_timeout_s': series_timeout_s,
#             'merge_gap_s': merge_gap_s,
#             'min_snr': min_snr
#         },
#         'spike_version': spike_version,
#     }
    
#     data['qc_metrics'] = qc_metrics
    
#     # Enhanced summary statistics
#     total_ca_events = ca_event_counts.sum()
#     mean_ca_rate = ca_event_rates.mean()
#     mean_ca_amplitude = ca_mean_amplitudes[ca_event_counts > 0].mean() if np.any(ca_event_counts > 0) else 0.0
#     mean_activity_fraction = qc_metrics['ca_activity_fraction'].mean()
    
#     print(f"Enhanced Ca transient detection results:")
#     print(f"  Total Ca events detected: {total_ca_events}")
#     print(f"  Mean Ca event rate: {mean_ca_rate:.3f} events/s")
#     print(f"  Mean Ca amplitude: {mean_ca_amplitude:.3f} ΔF/F")
#     print(f"  Mean activity fraction: {100*mean_activity_fraction:.1f}%")
#     print(f"  ROIs with events: {np.sum(ca_event_counts > 0)}/{N} ({100*np.sum(ca_event_counts > 0)/N:.1f}%)")
    
#     if spike_version is not None:
#         max_spike_rate = spike_metrics['spike_rate_mean_hz'].max()
#         print(f"Spike detection results:")
#         print(f"  Max spike rate: {max_spike_rate:.1f} Hz")
#         print(f"  Total spike onsets: {spike_metrics['spike_count_onsets'].sum()}")
    
#     print("Enhanced QC metrics computation complete")

def compute_spike_cv_isi_from_peaks(spike_peaks, fs):
    """Compute CV of ISI from detected peaks"""
    N = len(spike_peaks)
    cv_isi = np.zeros(N, dtype=np.float32)
    
    for i in range(N):
        peaks = spike_peaks[i]
        if len(peaks) < 3:
            cv_isi[i] = np.nan
            continue
        
        # Convert peak indices to times and compute ISI
        peak_times = peaks / fs
        isi = np.diff(peak_times)
        
        if len(isi) < 2:
            cv_isi[i] = np.nan
            continue
        
        mean_isi = np.mean(isi)
        std_isi = np.std(isi)
        
        if mean_isi > 0:
            cv_isi[i] = std_isi / mean_isi
        else:
            cv_isi[i] = np.nan
    
    return cv_isi

def spike_onset_mask(sp, k_mad=3.0):
    """Detect spike onsets from OASIS amplitude traces using robust thresholding"""
    sp = np.maximum(0, np.asarray(sp))              # no negatives
    mad = np.median(np.abs(sp - np.median(sp))) + 1e-12
    thr = k_mad * mad / 0.6745                      # robust sigma
    on = sp > max(1e-6, thr)
    onsets = on & np.concatenate([np.zeros((on.shape[0],1), bool), ~on[:, :-1]], axis=1)
    return onsets

def spike_rate_hz(spks, fs, bin_s, k_mad=3.0):
    """Convert OASIS spikes to firing rate in Hz using onset detection"""
    onsets = spike_onset_mask(spks, k_mad=k_mad)
    bin_len = max(1, int(round(bin_s * fs)))
    pad = (-onsets.shape[1]) % bin_len
    if pad: 
        onsets = np.pad(onsets, ((0,0),(0,pad)))
    counts = onsets.reshape(onsets.shape[0], -1, bin_len).sum(axis=2)
    return counts / bin_s  # Hz

def compute_spike_cv_isi(spike_onsets, fs):
    """Compute coefficient of variation of inter-spike intervals per ROI"""
    N = spike_onsets.shape[0]
    cv_isi = np.zeros(N, dtype=np.float32)
    
    for i in range(N):
        spike_times = np.where(spike_onsets[i])[0] / fs  # Convert to seconds
        
        if len(spike_times) < 3:  # Need at least 3 spikes for meaningful CV
            cv_isi[i] = np.nan
            continue
        
        # Compute inter-spike intervals
        isi = np.diff(spike_times)
        
        if len(isi) < 2:
            cv_isi[i] = np.nan
            continue
        
        # CV = std/mean
        mean_isi = np.mean(isi)
        std_isi = np.std(isi)
        
        if mean_isi > 0:
            cv_isi[i] = std_isi / mean_isi
        else:
            cv_isi[i] = np.nan
    
    return cv_isi

def estimate_signal_noise_ratio(data):
    """Estimate signal-to-noise ratio from dF/F traces"""
    if 'dFF_clean' not in data:
        print("    Warning: No dFF_clean found, using dFF for SNR estimation")
        dff = data.get('dFF')
    else:
        dff = data['dFF_clean']
    
    if dff is None:
        print("    Warning: No dF/F data found for SNR estimation")
        return np.zeros(1)
    
    N, T = dff.shape
    snr = np.zeros(N, dtype=np.float32)
    
    for i in range(N):
        trace = dff[i]
        
        # Remove NaN values
        valid_mask = np.isfinite(trace)
        if not np.any(valid_mask):
            snr[i] = np.nan
            continue
        
        trace_clean = trace[valid_mask]
        
        # Signal: standard deviation of the trace
        signal_power = np.std(trace_clean)
        
        # Noise: estimate from high-frequency components
        # Use difference between consecutive points as noise proxy
        if len(trace_clean) > 1:
            noise_proxy = np.diff(trace_clean)
            noise_power = np.std(noise_proxy) / np.sqrt(2)  # Adjust for differencing
        else:
            noise_power = 0.0
        
        # SNR calculation
        if noise_power > 1e-9:
            snr[i] = signal_power / noise_power
        else:
            snr[i] = np.inf if signal_power > 1e-9 else 1.0
    
    return snr

def compute_temporal_stability(data):
    """Compute temporal stability metrics (drift, stationarity)"""
    if 'dFF_clean' not in data:
        print("    Warning: No dFF_clean found, using dFF for stability estimation")
        dff = data.get('dFF')
    else:
        dff = data['dFF_clean']
    
    if dff is None:
        print("    Warning: No dF/F data found for stability estimation")
        return np.zeros(1)
    
    N, T = dff.shape
    stability = np.zeros(N, dtype=np.float32)
    
    # Divide trace into segments to measure drift
    n_segments = min(10, T // 100)  # At least 100 points per segment
    if n_segments < 2:
        # Too short for meaningful stability analysis
        return np.full(N, np.nan)
    
    segment_size = T // n_segments
    
    for i in range(N):
        trace = dff[i]
        
        # Remove NaN values
        valid_mask = np.isfinite(trace)
        if not np.any(valid_mask):
            stability[i] = np.nan
            continue
        
        # Calculate mean of each segment
        segment_means = []
        for seg in range(n_segments):
            start_idx = seg * segment_size
            end_idx = min((seg + 1) * segment_size, T)
            
            segment = trace[start_idx:end_idx]
            valid_segment = segment[np.isfinite(segment)]
            
            if len(valid_segment) > 0:
                segment_means.append(np.mean(valid_segment))
        
        if len(segment_means) < 2:
            stability[i] = np.nan
            continue
        
        # Stability = 1 / (coefficient of variation of segment means)
        # Lower CV means more stable
        segment_means = np.array(segment_means)
        mean_of_means = np.mean(segment_means)
        std_of_means = np.std(segment_means)
        
        if mean_of_means != 0:
            cv_segments = std_of_means / abs(mean_of_means)
            stability[i] = 1.0 / (1.0 + cv_segments)  # Maps to [0,1], higher = more stable
        else:
            stability[i] = 1.0  # Perfectly stable (zero signal)
    
    return stability





def detect_ca_transients_debug(x, fs, roi_id=None, debug=True, **params):
    """
    Debug version of Ca transient detection with concise logging
    """
    if debug and roi_id is not None:
        print(f"\n=== DEBUGGING ROI {roi_id} ===")
    
    x = np.asarray(x, float)
    T = x.size
    
    if debug:
        print(f"Input: {T} frames, fs={fs:.1f}Hz, duration={T/fs:.1f}s")
        print(f"dF/F range: [{np.nanmin(x):.3f}, {np.nanmax(x):.3f}], std={np.nanstd(x):.3f}")
    
    # Extract parameters with defaults
    smooth_sigma_s = params.get('smooth_sigma_s', 0.05)
    k_on = params.get('k_on', 1.5)
    k_off = params.get('k_off', 0.5)
    min_amp = params.get('min_amp', 0.05)
    min_series_peak_dff = params.get('min_series_peak_dff', 0.10)
    
    if debug:
        print(f"Parameters: k_on={k_on}, k_off={k_off}, min_amp={min_amp}")

    if T < 2:
        if debug: print("REJECT: Too few frames")
        return [], np.zeros(T, dtype=bool), {}
    
    # Check for invalid traces
    if np.all(np.isnan(x)) or np.all(x == x[0]):
        if debug: print("REJECT: Invalid trace (all NaN or constant)")
        return [], np.zeros(T, dtype=bool), {}

    # Smoothing
    if smooth_sigma_s > 0:
        sigma_frames = max(1, int(round(smooth_sigma_s * fs)))
        x_s = gaussian_filter1d(x, sigma=sigma_frames)
        if debug: print(f"Smoothed with sigma={sigma_frames} frames")
    else:
        x_s = x.copy()

    # Noise estimation
    try:
        valid_mask = np.isfinite(x_s)
        if not np.any(valid_mask):
            if debug: print("REJECT: No valid data points")
            return [], np.zeros(T, dtype=bool), {}
        
        x_valid = x_s[valid_mask]
        
        if len(x_valid) < 10:
            if debug: print("REJECT: Too few valid points")
            return [], np.zeros(T, dtype=bool), {}
        
        # Simple noise estimation for debugging
        p10 = np.percentile(x_valid, 10)
        p90 = np.percentile(x_valid, 90)
        median_val = np.median(x_valid)
        
        # Use MAD for noise estimate
        mad = np.median(np.abs(x_valid - median_val))
        if mad < 1e-12:
            mad = np.std(x_valid) / 1.4826
            if mad < 1e-12:
                if debug: print("REJECT: Zero variance trace")
                return [], np.zeros(T, dtype=bool), {}
        
        sigma = 1.4826 * mad
        
        if debug:
            print(f"Noise estimation: median={median_val:.3f}, sigma={sigma:.3f}, mad={mad:.3f}")
        
    except Exception as e:
        if debug: print(f"REJECT: Noise estimation error: {e}")
        return [], np.zeros(T, dtype=bool), {}

    # Thresholds
    y = x_s
    th_on = median_val + k_on * sigma
    th_off = median_val + k_off * sigma
    
    if debug:
        print(f"Thresholds: th_on={th_on:.3f}, th_off={th_off:.3f}")
        above_on = np.sum(y > th_on)
        above_off = np.sum(y > th_off)
        print(f"Points above thresholds: {above_on} above th_on, {above_off} above th_off")

    # Track detection stages
    debug_info = {
        'potential_onsets': [],
        'rejected_reasons': {'no_peak': 0, 'low_amp': 0, 'no_offset': 0},
        'final_events': []
    }
    
    events = []
    mask = np.zeros(T, dtype=bool)
    i = 1
    onset_count = 0
    
    # Main detection loop - CONCISE VERSION
    while i < T:
        if not np.isfinite(y[i-1]) or not np.isfinite(y[i]):
            i += 1
            continue
            
        # ONSET detection
        if y[i-1] < th_on and y[i] >= th_on:
            onset_count += 1
            i_on = i
            
            # Find baseline
            j = i_on
            while j > 0 and np.isfinite(y[j-1]) and y[j-1] > th_off:
                j -= 1
            i_base = j
            baseline = y[i_base]
            
            # Quick amplitude check
            search_end = min(T, i_on + int(2.0 * fs))  # 2 second search
            if search_end > i_on:
                peak_val = np.nanmax(y[i_on:search_end])
                amp = peak_val - baseline
                
                if amp >= min_amp and amp >= min_series_peak_dff:
                    # For debugging, create a simple event
                    i_peak = i_on + np.nanargmax(y[i_on:search_end])
                    
                    # Find offset (simplified)
                    i_off = None
                    for k in range(i_peak, min(T, i_peak + int(6.0 * fs))):
                        if y[k] < th_off:
                            i_off = k
                            break
                    
                    if i_off is not None:
                        events.append({
                            't_on': i_on/fs, 't_peak': i_peak/fs, 't_off': i_off/fs,
                            'amp': amp, 'i_on': i_on, 'i_peak': i_peak, 'i_off': i_off
                        })
                        mask[i_base:i_off] = True
                        i = i_off + 1
                        continue
                    else:
                        debug_info['rejected_reasons']['no_offset'] += 1
                else:
                    debug_info['rejected_reasons']['low_amp'] += 1
            else:
                debug_info['rejected_reasons']['no_peak'] += 1
            
            i += 1
        else:
            i += 1
    
    if debug:
        print(f"\nCONCISE SUMMARY:")
        print(f"  Found {onset_count} potential onsets -> {len(events)} final events")
        print(f"  Rejection reasons:")
        for reason, count in debug_info['rejected_reasons'].items():
            print(f"    {reason}: {count}")
        
        if len(events) > 0:
            amps = [e['amp'] for e in events]
            times = [e['t_peak'] for e in events]
            print(f"  Event amplitudes: {np.min(amps):.3f} - {np.max(amps):.3f} ΔF/F")
            print(f"  Event times: {np.min(times):.1f} - {np.max(times):.1f} s")
            print(f"  Mean amplitude: {np.mean(amps):.3f} ΔF/F")
            print(f"  Event rate: {len(events)/(T/fs):.3f} events/s")
            
            # Show only first few and last few events
            if len(events) <= 10:
                print(f"  All events:")
                for i, e in enumerate(events):
                    print(f"    {i+1}: t={e['t_peak']:.1f}s, amp={e['amp']:.3f}")
            else:
                print(f"  Sample events (first 3, last 3):")
                for i in [0, 1, 2]:
                    e = events[i]
                    print(f"    {i+1}: t={e['t_peak']:.1f}s, amp={e['amp']:.3f}")
                print(f"    ... ({len(events)-6} more events) ...")
                for i in range(len(events)-3, len(events)):
                    e = events[i]
                    print(f"    {i+1}: t={e['t_peak']:.1f}s, amp={e['amp']:.3f}")
        else:
            print(f"  No events detected")
    
    debug_info['final_events'] = events
    return events, mask, debug_info


def debug_ca_detection_with_visualization(data: Dict[str, Any], cfg: Dict[str, Any], roi: int, save: bool = False) -> Any:
    """Debug Ca²⁺ transient detection with full visualization of thresholds and rejections"""
    print(f"\n=== ENHANCED Ca²⁺ DETECTION DEBUG - ROI {roi} ===")
    
    if "dFF_clean" not in data:
        raise ValueError("Missing dFF_clean - run processing first")
    
    rv = cfg['review']
    segments = list(rv['detail_segments'])
    fs = float(cfg['acq']['fs'])
    
    ca_cfg = cfg.get("ca_transients", {})
    
    N, T = data["dFF_clean"].shape
    if roi < 0 or roi >= N:
        raise IndexError(f"ROI {roi} out of range (N={N})")
    
    detail_idx = _segments_to_idx(segments, fs, T)
    n_segments = len(detail_idx)
    
    # Create comprehensive debug figure
    fig, axes = plt.subplots(n_segments, 1, figsize=(18, 5*n_segments), sharex=False)
    if n_segments == 1:
        axes = [axes]
    
    total_debug_events = 0
    
    for seg_idx, (slc, t_local, (sa, sb)) in enumerate(detail_idx):
        ax = axes[seg_idx]
        
        print(f"\n--- SEGMENT {seg_idx+1}: {sa:.1f}-{sb:.1f}s ---")
        
        # Extract segment data
        dff_segment = data['dFF_clean'][roi, slc]
        
        # Run enhanced debug detection
        events, mask, debug_info = detect_ca_transients_debug_enhanced(
            dff_segment, fs, roi_id=f"{roi}_seg{seg_idx+1}", debug=True, **ca_cfg
        )
        
        total_debug_events += len(events)
        
        # Plot the raw trace
        ax.plot(t_local, dff_segment, 'k-', linewidth=1.0, alpha=0.8, label='dF/F')
        
        # Plot smoothed trace if different
        if 'smoothed_trace' in debug_info:
            smoothed = debug_info['smoothed_trace']
            if not np.array_equal(smoothed, dff_segment):
                ax.plot(t_local, smoothed, 'gray', linewidth=0.8, alpha=0.6, label='Smoothed')
        
        # Show thresholds
        th_on = debug_info['th_on']
        th_off = debug_info['th_off']
        baseline = debug_info['baseline']
        
        ax.axhline(th_on, color='red', linestyle='--', alpha=0.7, linewidth=1, label=f'Onset threshold ({th_on:.3f})')
        ax.axhline(th_off, color='orange', linestyle='--', alpha=0.7, linewidth=1, label=f'Offset threshold ({th_off:.3f})')
        ax.axhline(baseline, color='blue', linestyle='-', alpha=0.5, linewidth=1, label=f'Baseline ({baseline:.3f})')
        
        # Mark detected events (ACCEPTED)
        for i, event in enumerate(events):
            t_peak_abs = t_local[0] + event['t_peak'] * fs  # Convert back to absolute time
            t_on_abs = t_local[0] + event['t_on'] * fs
            t_off_abs = t_local[0] + event['t_off'] * fs
            
            # Find indices in original segment
            peak_rel_idx = int(round(event['t_peak'] * fs))
            on_rel_idx = int(round(event['t_on'] * fs))
            off_rel_idx = int(round(event['t_off'] * fs))
            
            if 0 <= peak_rel_idx < len(dff_segment):
                # Accepted event span
                if 0 <= on_rel_idx < len(t_local) and 0 <= off_rel_idx < len(t_local):
                    ax.axvspan(t_local[on_rel_idx], t_local[off_rel_idx], 
                             alpha=0.2, color='green', zorder=1, label='Accepted event' if i == 0 else "")
                
                # Peak marker
                ax.scatter(t_local[peak_rel_idx], dff_segment[peak_rel_idx], 
                         c='green', s=50, marker='^', zorder=10, 
                         edgecolors='darkgreen', linewidth=1)
                
                # Amplitude annotation
                ax.annotate(f'{event["amp"]:.2f}', 
                           xy=(t_local[peak_rel_idx], dff_segment[peak_rel_idx]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='green', weight='bold')
        
        # Mark REJECTED onsets
        for rejection in debug_info['rejection_details']:
            reason = rejection['reason']
            i_on_abs = rejection['i_on']
            i_on_rel = i_on_abs - slc.start  # Convert to segment-relative index
            
            if 0 <= i_on_rel < len(t_local):
                color_map = {'low_amp': 'red', 'no_offset': 'purple', 'no_peak': 'orange'}
                color = color_map.get(reason, 'gray')
                
                # Mark rejected onset
                ax.scatter(t_local[i_on_rel], dff_segment[i_on_rel], 
                         c=color, s=30, marker='x', zorder=9, alpha=0.7,
                         label=f'Rejected: {reason}' if rejection == debug_info['rejection_details'][0] else "")
                
                # Add reason annotation
                ax.annotate(f'{reason}\n{rejection.get("amp", 0):.2f}', 
                           xy=(t_local[i_on_rel], dff_segment[i_on_rel]),
                           xytext=(3, -15), textcoords='offset points',
                           fontsize=7, color=color, alpha=0.8)
        
        # Add detection statistics to title
        n_accepted = len(events)
        n_rejected = sum(debug_info['rejected_reasons'].values())
        n_total_onsets = n_accepted + n_rejected
        
        title = f'ROI {roi} - Segment {seg_idx+1} ({sa:.1f}-{sb:.1f}s): {n_accepted} accepted, {n_rejected} rejected (of {n_total_onsets} onsets)'
        ax.set_title(title, fontsize=10)
        
        ax.set_ylabel('dF/F')
        ax.set_xlim(sa, sb)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    
    # Compare with regular QC results
    print(f"\n=== COMPARISON WITH QC METRICS ===")
    print(f"Total debug events across segments: {total_debug_events}")
    
    qc = data.get('qc_metrics', {})
    if 'ca_events' in qc and len(qc['ca_events']) > roi:
        qc_events = qc['ca_events'][roi]
        print(f"QC metrics total events for ROI: {len(qc_events)}")
        
        # Count QC events in our analyzed segments
        qc_events_in_segments = 0
        for slc, t_local, (sa, sb) in detail_idx:
            seg_events = [e for e in qc_events if sa <= e['t_peak'] <= sb]
            qc_events_in_segments += len(seg_events)
        
        print(f"QC events in analyzed segments: {qc_events_in_segments}")
        
        if qc_events_in_segments != total_debug_events:
            print(f"*** MISMATCH DETECTED ***")
            print(f"  Debug algorithm: {total_debug_events} events")
            print(f"  QC algorithm: {qc_events_in_segments} events")
            print(f"  Difference: {total_debug_events - qc_events_in_segments}")
    else:
        print("No QC events found for comparison")
    
    if save:
        save_dir = cfg['overlay']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'ca_debug_detailed_roi_{roi:04d}.png')
        fig.savefig(save_path, dpi=cfg['overlay']['dpi'], bbox_inches='tight')
        print(f"Detailed debug plot saved: {save_path}")
    
    plt.show()
    return fig

def debug_ca_detection_in_segments(data: Dict[str, Any], cfg: Dict[str, Any], roi: int, save: bool = False) -> Any:
    """Debug Ca²⁺ transient detection in the same segments used for review plots"""
    print(f"\n=== DEBUGGING Ca²⁺ DETECTION IN SEGMENTS - ROI {roi} ===")
    
    if "dFF_clean" not in data:
        raise ValueError("Missing dFF_clean - run processing first")
    
    rv = cfg['review']
    segments = list(rv['detail_segments'])
    fs = float(cfg['acq']['fs'])
    
    ca_cfg = cfg.get("ca_transients", {})
    print(f"Using Ca detection parameters: {ca_cfg}")
    
    N, T = data["dFF_clean"].shape
    if roi < 0 or roi >= N:
        raise IndexError(f"ROI {roi} out of range (N={N})")
    
    detail_idx = _segments_to_idx(segments, fs, T)
    n_segments = len(detail_idx)
    
    print(f"Analyzing {n_segments} segments for ROI {roi}")
    
    # Create figure with subplots for each segment
    fig, axes = plt.subplots(n_segments, 1, figsize=(16, 4*n_segments), sharex=False)
    if n_segments == 1:
        axes = [axes]
    
    segment_summaries = []
    
    for seg_idx, (slc, t_local, (sa, sb)) in enumerate(detail_idx):
        ax = axes[seg_idx]
        
        print(f"\n--- SEGMENT {seg_idx+1}: {sa:.1f}-{sb:.1f}s ---")
        
        # Extract segment data
        dff_segment = data['dFF_clean'][roi, slc]
        
        # Run debug detection on this segment only
        print(f"Running detection on segment: {len(dff_segment)} points, {len(t_local)} time points")
        
        events, mask, debug_info = detect_ca_transients_debug(
            dff_segment, fs, roi_id=f"{roi}_seg{seg_idx+1}", debug=True, **ca_cfg
        )
        
        # Plot the trace
        ax.plot(t_local, dff_segment, 'k-', linewidth=0.8, alpha=0.7, label='dF/F')
        
        # Show thresholds if available from debug
        if debug_info:
            # We need to extract threshold info from the debug function
            # Let's modify the debug function to return this info
            pass
        
        # Mark detected events
        if len(events) > 0:
            for i, event in enumerate(events):
                # Convert event times back to absolute time
                t_peak_abs = sa + event['t_peak']
                t_on_abs = sa + event['t_on'] 
                t_off_abs = sa + event['t_off']
                
                # Find corresponding indices in the segment
                peak_idx = int(round(event['t_peak'] * fs))
                on_idx = int(round(event['t_on'] * fs))
                off_idx = int(round(event['t_off'] * fs))
                
                if 0 <= peak_idx < len(dff_segment):
                    # Peak marker
                    ax.scatter(t_peak_abs, dff_segment[peak_idx], 
                             c='red', s=40, marker='^', zorder=10, 
                             edgecolors='darkred', linewidth=1)
                    
                    # Event span
                    if 0 <= on_idx < len(dff_segment) and 0 <= off_idx < len(dff_segment):
                        ax.axvspan(t_on_abs, t_off_abs, alpha=0.2, color='orange', zorder=1)
                        
                        # Amplitude line
                        baseline_val = event.get('baseline', dff_segment[on_idx])
                        ax.plot([t_peak_abs, t_peak_abs], 
                               [baseline_val, dff_segment[peak_idx]], 
                               'r--', alpha=0.7, linewidth=1)
                        
                        # Amplitude text
                        ax.text(t_peak_abs, dff_segment[peak_idx] + 0.1, 
                               f'{event["amp"]:.2f}', 
                               ha='center', va='bottom', fontsize=8, color='red')
        
        # Add threshold lines and rejection info
        # This requires enhancing the debug function to return threshold values
        
        # Title with detection summary
        n_events = len(events)
        if n_events > 0:
            total_amp = sum(e['amp'] for e in events)
            mean_amp = total_amp / n_events
            ax.set_title(f'Segment {seg_idx+1} ({sa:.1f}-{sb:.1f}s): {n_events} events detected, mean amp={mean_amp:.2f}')
        else:
            ax.set_title(f'Segment {seg_idx+1} ({sa:.1f}-{sb:.1f}s): No events detected')
        
        ax.set_ylabel('dF/F')
        ax.set_xlim(sa, sb)
        ax.grid(True, alpha=0.3)
        
        # Store segment summary
        segment_summaries.append({
            'segment': seg_idx + 1,
            'time_range': (sa, sb),
            'n_events': n_events,
            'events': events,
            'debug_info': debug_info
        })
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    
    # Print detailed summary
    print(f"\n=== SEGMENT-BY-SEGMENT SUMMARY ===")
    total_events_all_segments = 0
    for summary in segment_summaries:
        seg_num = summary['segment']
        sa, sb = summary['time_range']
        n_events = summary['n_events']
        total_events_all_segments += n_events
        
        print(f"Segment {seg_num} ({sa:.1f}-{sb:.1f}s): {n_events} events")
        
        if n_events > 0:
            events = summary['events']
            for i, event in enumerate(events):
                print(f"  Event {i+1}: t={sa + event['t_peak']:.1f}s, amp={event['amp']:.3f}")
    
    print(f"\nTotal events across all segments: {total_events_all_segments}")
    
    # Compare with QC metrics if available
    qc = data.get('qc_metrics', {})
    if 'ca_events' in qc and len(qc['ca_events']) > roi:
        qc_events = qc['ca_events'][roi]
        print(f"QC metrics reports: {len(qc_events)} events for this ROI")
        
        # Show which QC events fall in our segments
        qc_events_in_segments = []
        for seg_summary in segment_summaries:
            sa, sb = seg_summary['time_range']
            seg_qc_events = [e for e in qc_events if sa <= e['t_peak'] <= sb]
            qc_events_in_segments.extend(seg_qc_events)
            if len(seg_qc_events) > 0:
                print(f"  QC events in segment {seg_summary['segment']}: {len(seg_qc_events)}")
        
        print(f"Total QC events in analyzed segments: {len(qc_events_in_segments)}")
        
        if len(qc_events_in_segments) != total_events_all_segments:
            print(f"*** MISMATCH: Debug found {total_events_all_segments}, QC found {len(qc_events_in_segments)} in same segments ***")
    
    if save:
        save_dir = cfg['overlay']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'ca_debug_segments_roi_{roi:04d}.png')
        fig.savefig(save_path, dpi=cfg['overlay']['dpi'], bbox_inches='tight')
        print(f"Segment debug plot saved: {save_path}")
    
    plt.show()
    
    return fig, segment_summaries





def detect_ca_transients_debug_enhanced(x, fs, roi_id=None, debug=True, **params):
    """
    Enhanced debug version that uses ALL config parameters and returns threshold information
    """
    if debug and roi_id is not None:
        print(f"\n=== DEBUGGING {roi_id} ===")
    
    x = np.asarray(x, float)
    T = x.size
    
    if debug:
        print(f"Input: {T} frames, fs={fs:.1f}Hz, duration={T/fs:.1f}s")
        print(f"dF/F range: [{np.nanmin(x):.3f}, {np.nanmax(x):.3f}], std={np.nanstd(x):.3f}")
    
    # Extract ALL parameters from config - no defaults here
    required_params = [
        'smooth_sigma_s', 'k_on', 'k_off', 'rise_max_s', 'peak_win_s', 'decay_max_s',
        'min_amp', 'min_series_peak_dff', 'series_timeout_s', 'merge_gap_s', 'min_snr',
        'max_decay_slope_per_s', 'min_event_duration_s', 'max_baseline_drift_per_s', 'require_clear_peak'
    ]
    
    for param in required_params:
        if param not in params:
            raise ValueError(f"Missing required parameter: {param}. Check your config.yaml ca_transients section.")
    
    # Basic parameters
    smooth_sigma_s = params['smooth_sigma_s']
    k_on = params['k_on']
    k_off = params['k_off']
    min_amp = params['min_amp']
    
    # Kinetic constraints
    rise_max_s = params['rise_max_s']
    peak_win_s = params['peak_win_s']
    decay_max_s = params['decay_max_s']
    
    # Biological constraints
    max_decay_slope_per_s = params['max_decay_slope_per_s']
    min_series_peak_dff = params['min_series_peak_dff']
    series_timeout_s = params['series_timeout_s']
    merge_gap_s = params['merge_gap_s']
    min_snr = params['min_snr']
    min_event_duration_s = params['min_event_duration_s']
    max_baseline_drift_per_s = params['max_baseline_drift_per_s']
    require_clear_peak = params['require_clear_peak']
    
    if debug:
        print(f"Config parameters (ALL from YAML):")
        print(f"  Basic: k_on={k_on}, k_off={k_off}, min_amp={min_amp}, smooth_sigma_s={smooth_sigma_s}")
        print(f"  Kinetics: rise_max_s={rise_max_s}, peak_win_s={peak_win_s}, decay_max_s={decay_max_s}")
        print(f"  Biological: max_decay_slope_per_s={max_decay_slope_per_s}, min_series_peak_dff={min_series_peak_dff}")
        print(f"  Detection: series_timeout_s={series_timeout_s}, merge_gap_s={merge_gap_s}, min_snr={min_snr}")
        print(f"  Quality: min_event_duration_s={min_event_duration_s}, max_baseline_drift_per_s={max_baseline_drift_per_s}")
        print(f"  Shape: require_clear_peak={require_clear_peak}")

    if T < 2:
        if debug: print("REJECT: Too few frames")
        return [], np.zeros(T, dtype=bool), {'th_on': 0, 'th_off': 0, 'baseline': 0, 'sigma': 0}
    
    # Check for invalid traces
    if np.all(np.isnan(x)) or np.all(x == x[0]):
        if debug: print("REJECT: Invalid trace (all NaN or constant)")
        return [], np.zeros(T, dtype=bool), {'th_on': 0, 'th_off': 0, 'baseline': 0, 'sigma': 0}

    # Smoothing (using config parameter)
    if smooth_sigma_s > 0:
        sigma_frames = max(1, int(round(smooth_sigma_s * fs)))
        x_s = gaussian_filter1d(x, sigma=sigma_frames)
        if debug: print(f"Smoothed with sigma={sigma_frames} frames")
    else:
        x_s = x.copy()

    # IMPROVED: More robust noise estimation (same as main function)
    try:
        valid_mask = np.isfinite(x_s)
        if not np.any(valid_mask):
            if debug: print("REJECT: No valid data points")
            return [], np.zeros(T, dtype=bool), {'th_on': 0, 'th_off': 0, 'baseline': 0, 'sigma': 0}
        
        x_valid = x_s[valid_mask]
        
        if len(x_valid) < 10 or np.std(x_valid) < 1e-12:
            if debug: print("REJECT: Too few valid points or zero variance")
            return [], np.zeros(T, dtype=bool), {'th_on': 0, 'th_off': 0, 'baseline': 0, 'sigma': 0}
        
        # Use multiple percentiles for more robust baseline estimation
        p10 = np.percentile(x_valid, 10)
        p20 = np.percentile(x_valid, 20) 
        p30 = np.percentile(x_valid, 30)
        
        # Use the most stable baseline region
        baseline_candidates = [p10, p20, p30]
        residuals = []
        for p in baseline_candidates:
            r = x_valid - p
            mad = np.median(np.abs(r - np.median(r)))
            residuals.append(mad)
        
        # Choose baseline with smallest MAD (most stable)
        best_baseline = baseline_candidates[np.argmin(residuals)]
        r = x_valid - best_baseline
        mad = np.median(np.abs(r - np.median(r)))
        
        if mad < 1e-12:
            mad = np.std(x_valid) / 1.4826
            if mad < 1e-12:
                if debug: print("REJECT: Zero variance trace")
                return [], np.zeros(T, dtype=bool), {'th_on': 0, 'th_off': 0, 'baseline': best_baseline, 'sigma': 0}
        
        sigma = 1.4826 * mad
        
        if debug:
            print(f"Noise estimation: baseline={best_baseline:.3f}, sigma={sigma:.3f}, mad={mad:.3f}")
        
    except Exception as e:
        if debug: print(f"REJECT: Noise estimation error: {e}")
        return [], np.zeros(T, dtype=bool), {'th_on': 0, 'th_off': 0, 'baseline': 0, 'sigma': 0}

    # Adaptive thresholds based on signal characteristics
    try:
        # Use the robust baseline estimate
        med_val = best_baseline
        y = x_s
        th_on = med_val + k_on * sigma
        th_off = med_val + k_off * sigma
        amp_thr = min_amp
            
    except Exception as e:
        if debug: print(f"REJECT: Threshold calculation error: {e}")
        return [], np.zeros(T, dtype=bool), {'th_on': 0, 'th_off': 0, 'baseline': 0, 'sigma': 0}

    # Convert time parameters to samples
    rise_max = int(round(rise_max_s * fs))
    peak_win = int(round(peak_win_s * fs))
    decay_max = int(round(decay_max_s * fs))
    series_timeout = int(round(series_timeout_s * fs))
    merge_gap = int(round(merge_gap_s * fs))
    min_duration = int(round(min_event_duration_s * fs))
    
    if debug:
        print(f"Thresholds: th_on={th_on:.3f}, th_off={th_off:.3f}")
        above_on = np.sum(y > th_on)
        above_off = np.sum(y > th_off)
        print(f"Points above thresholds: {above_on} above th_on, {above_off} above th_off")

    # Enhanced tracking of detection stages
    debug_info = {
        'th_on': th_on,
        'th_off': th_off, 
        'baseline': med_val,
        'sigma': sigma,
        'smoothed_trace': y.copy(),
        'potential_onsets': [],
        'rejected_reasons': {'no_peak': 0, 'low_amp': 0, 'no_offset': 0, 'rise_too_slow': 0, 
                           'decay_too_fast': 0, 'short_duration': 0, 'low_snr': 0, 
                           'baseline_drift': 0, 'weak_peak': 0},
        'rejection_details': [],  # Store details for each rejection
        'final_events': []
    }
    
    events = []
    mask = np.zeros(T, dtype=bool)
    i = 1
    onset_count = 0
    
    # Main detection loop with detailed rejection tracking using ALL CONFIG PARAMETERS
    try:
        while i < T:
            # Skip NaN values
            if not np.isfinite(y[i-1]) or not np.isfinite(y[i]):
                i += 1
                continue
                
            # ONSET detection
            if y[i-1] < th_on and y[i] >= th_on:
                onset_count += 1
                i_on = i
                
                # Store potential onset
                debug_info['potential_onsets'].append({
                    'i_on': i_on,
                    't_on': i_on/fs,
                    'value': y[i_on]
                })
                
                # Find baseline (last crossing below th_off)
                j = i_on
                while j > 0 and np.isfinite(y[j-1]) and y[j-1] > th_off:
                    j -= 1
                i_base = j
                
                if not np.isfinite(y[i_base]):
                    i += 1
                    continue
                    
                baseline = y[i_base]

                # Rise time check (using config parameter)
                j = i_on
                while j > 0 and np.isfinite(y[j-1]) and y[j-1] > th_off:
                    j -= 1
                rise_len = i_on - j
                if rise_len > rise_max:
                    debug_info['rejected_reasons']['rise_too_slow'] += 1
                    debug_info['rejection_details'].append({
                        'reason': 'rise_too_slow',
                        'i_on': i_on, 't_on': i_on/fs,
                        'rise_len': rise_len, 'rise_max': rise_max
                    })
                    i += 1
                    continue

                # Peak search in event series (using config parameter)
                i_end_search = min(T, i_on + series_timeout)
                max_peak_in_series = np.nanmax(y[i_on:i_end_search]) if i_end_search > i_on else y[i_on]
                series_amp = max_peak_in_series - baseline
                
                if series_amp < min_series_peak_dff:
                    debug_info['rejected_reasons']['low_amp'] += 1
                    debug_info['rejection_details'].append({
                        'reason': 'low_series_peak',
                        'i_on': i_on, 't_on': i_on/fs,
                        'series_amp': series_amp, 'min_required': min_series_peak_dff
                    })
                    i += 1
                    continue

                # Find peak within initial window (using config parameter)
                i_end_peak = min(T, i_on + max(peak_win, 1))
                peak_segment = y[i_on:i_end_peak]
                valid_peak_mask = np.isfinite(peak_segment)
                if not np.any(valid_peak_mask):
                    debug_info['rejected_reasons']['no_peak'] += 1
                    debug_info['rejection_details'].append({
                        'reason': 'no_valid_peak',
                        'i_on': i_on, 't_on': i_on/fs
                    })
                    i += 1
                    continue
                
                local_peak_idx = np.nanargmax(peak_segment)
                i_peak = i_on + local_peak_idx
                peak_val = y[i_peak]
                
                if not np.isfinite(peak_val):
                    debug_info['rejected_reasons']['no_peak'] += 1
                    debug_info['rejection_details'].append({
                        'reason': 'invalid_peak',
                        'i_on': i_on, 't_on': i_on/fs
                    })
                    i += 1
                    continue
                
                amp = peak_val - baseline
                if amp < amp_thr:
                    debug_info['rejected_reasons']['low_amp'] += 1
                    debug_info['rejection_details'].append({
                        'reason': 'low_amp',
                        'i_on': i_on, 't_on': i_on/fs,
                        'amp': amp, 'min_required': amp_thr
                    })
                    i += 1
                    continue

                # Find decay with merging allowed (using config parameter)
                i_search_end = min(T, i_on + max(decay_max, 1))
                k = i_peak
                i_off = None
                
                while k < i_search_end:
                    if not np.isfinite(y[k]):
                        k += 1
                        continue
                        
                    # Allow re-crossings above onset (complex events)
                    if y[k] >= th_on and y[k] > peak_val:
                        i_peak, peak_val = k, y[k]
                        amp = peak_val - baseline
                    
                    if y[k] < th_off:
                        i_off = k
                        break
                    k += 1

                if i_off is None:
                    debug_info['rejected_reasons']['no_offset'] += 1
                    debug_info['rejection_details'].append({
                        'reason': 'no_offset',
                        'i_on': i_on, 't_on': i_on/fs,
                        'amp': amp, 'baseline': baseline
                    })
                    i += 1
                    continue

                # Event duration check (using config parameter)
                event_duration_s = (i_off - i_base) / fs
                if event_duration_s < min_event_duration_s:
                    debug_info['rejected_reasons']['short_duration'] += 1
                    debug_info['rejection_details'].append({
                        'reason': 'short_duration',
                        'i_on': i_on, 't_on': i_on/fs,
                        'duration_s': event_duration_s, 'min_required': min_event_duration_s
                    })
                    i += 1
                    continue

                # Decay slope constraint (using config parameter)
                decay_s = (i_off - i_peak) / fs
                if decay_s > 0:
                    decay_slope = (peak_val - y[i_off]) / decay_s
                    if decay_slope > max_decay_slope_per_s:
                        debug_info['rejected_reasons']['decay_too_fast'] += 1
                        debug_info['rejection_details'].append({
                            'reason': 'decay_too_fast',
                            'i_on': i_on, 't_on': i_on/fs,
                            'decay_slope': decay_slope, 'max_allowed': max_decay_slope_per_s
                        })
                        i += 1
                        continue

                # Baseline stability check during event (using config parameter)
                if max_baseline_drift_per_s > 0:
                    event_segment = y[i_base:i_off]
                    if len(event_segment) > 3:
                        # Check if baseline drifts too much during event
                        baseline_trend = np.polyfit(np.arange(len(event_segment)), event_segment, 1)[0]
                        baseline_drift_rate = abs(baseline_trend * fs)
                        if baseline_drift_rate > max_baseline_drift_per_s:
                            debug_info['rejected_reasons']['baseline_drift'] += 1
                            debug_info['rejection_details'].append({
                                'reason': 'baseline_drift',
                                'i_on': i_on, 't_on': i_on/fs,
                                'drift_rate': baseline_drift_rate, 'max_allowed': max_baseline_drift_per_s
                            })
                            i += 1
                            continue

                # Peak prominence check (using config parameter)
                if require_clear_peak and len(event_segment) > 5:
                    # Require peak to be clearly above surrounding region
                    peak_rel_idx = i_peak - i_base
                    if peak_rel_idx > 2 and peak_rel_idx < len(event_segment) - 2:
                        surrounding = np.concatenate([
                            event_segment[max(0, peak_rel_idx-2):peak_rel_idx],
                            event_segment[peak_rel_idx+1:min(len(event_segment), peak_rel_idx+3)]
                        ])
                        if len(surrounding) > 0:
                            peak_prominence = peak_val - np.max(surrounding)
                            if peak_prominence < 0.3 * amp:  # Peak must be 30% above surroundings
                                debug_info['rejected_reasons']['weak_peak'] += 1
                                debug_info['rejection_details'].append({
                                    'reason': 'weak_peak',
                                    'i_on': i_on, 't_on': i_on/fs,
                                    'peak_prominence': peak_prominence, 'required': 0.3 * amp
                                })
                                i += 1
                                continue

                # SNR requirement (using config parameter)
                if min_snr > 0:
                    pre_event_start = max(0, i_base - 30)  # Shorter pre-event window
                    if i_base > pre_event_start:
                        noise_segment = y[pre_event_start:i_base]
                        noise_std = np.nanstd(noise_segment[np.isfinite(noise_segment)])
                        if noise_std > 0:
                            local_snr = amp / noise_std
                            if local_snr < min_snr:
                                debug_info['rejected_reasons']['low_snr'] += 1
                                debug_info['rejection_details'].append({
                                    'reason': 'low_snr',
                                    'i_on': i_on, 't_on': i_on/fs,
                                    'snr': local_snr, 'min_required': min_snr
                                })
                                i += 1
                                continue

                # Calculate event metrics
                rise_s = (i_peak - i_base) / fs
                decay_s = (i_off - i_peak) / fs
                
                # Calculate area
                event_segment = y[i_base:i_off]
                valid_event_mask = np.isfinite(event_segment)
                if np.any(valid_event_mask):
                    area_values = np.maximum(event_segment[valid_event_mask] - baseline, 0)
                    area = np.trapz(area_values, dx=1/fs)
                else:
                    area = 0.0

                # Event merging (using config parameter)
                if events and (i_on - events[-1]['i_off']) <= merge_gap:
                    # Merge with previous event
                    last_event = events[-1]
                    
                    # Update peak if current is higher
                    if peak_val > y[last_event['i_peak']]:
                        events[-1].update({
                            't_peak': i_peak/fs,
                            'i_peak': i_peak,
                            'amp': max(amp, last_event['amp'])
                        })
                    
                    # Extend off time
                    events[-1]['t_off'] = i_off/fs
                    events[-1]['i_off'] = i_off
                    events[-1]['decay_s'] = (i_off - events[-1]['i_peak']) / fs
                    
                    # Recalculate area
                    merged_segment = y[last_event['i_base']:i_off]
                    valid_merged_mask = np.isfinite(merged_segment)
                    if np.any(valid_merged_mask):
                        merged_area_values = np.maximum(merged_segment[valid_merged_mask] - last_event['baseline'], 0)
                        events[-1]['area'] = np.trapz(merged_area_values, dx=1/fs)
                    
                    # Update mask
                    mask[last_event['i_base']:i_off] = True
                    
                else:
                    # Create new event
                    event = {
                        't_on': i_base/fs,     # Absolute time of baseline/onset
                        't_peak': i_peak/fs,   # Absolute time of peak
                        't_off': i_off/fs,     # Absolute time of offset
                        'amp': amp, 
                        'rise_s': rise_s,
                        'decay_s': decay_s,
                        'area': area,
                        'baseline': baseline,
                        'i_on': i_on, 'i_peak': i_peak, 'i_off': i_off, 'i_base': i_base
                    }
                    events.append(event)
                    mask[i_base:i_off] = True

                i = i_off + 1
            else:
                i += 1

    except Exception as e:
        if debug: print(f"REJECT: Event detection loop error: {e}")
        pass
    
    debug_info['final_events'] = events
    
    if debug:
        print(f"\nENHANCED SUMMARY:")
        print(f"  Found {onset_count} potential onsets -> {len(events)} final events")
        print(f"  Rejection reasons:")
        for reason, count in debug_info['rejected_reasons'].items():
            if count > 0:
                print(f"    {reason}: {count}")
        
        if len(events) > 0:
            amps = [e['amp'] for e in events]
            times = [e['t_peak'] for e in events]
            print(f"  Event amplitudes: {np.min(amps):.3f} - {np.max(amps):.3f} ΔF/F")
            print(f"  Event times: {np.min(times):.1f} - {np.max(times):.1f} s")
            print(f"  Event rate: {len(events)/(T/fs):.3f} events/s")
            
            # Show sample events with enhanced metrics
            print(f"  Sample events (enhanced detection):")
            for i in range(min(3, len(events))):
                e = events[i]
                print(f"    Event {i+1}: t_on={e['t_on']:.2f}s, t_peak={e['t_peak']:.2f}s, t_off={e['t_off']:.2f}s")
                print(f"                amp={e['amp']:.3f}, rise={e['rise_s']:.2f}s, decay={e['decay_s']:.2f}s, area={e['area']:.2f}")
        else:
            print(f"  No events detected with current config parameters")
            # Show a few rejection examples with enhanced reasons
            if debug_info['rejection_details']:
                print(f"  Sample rejections:")
                for i, rejection in enumerate(debug_info['rejection_details'][:5]):
                    reason = rejection['reason']
                    print(f"    Rejection {i+1}: {reason} at t={rejection['t_on']:.2f}s")
                    if 'amp' in rejection:
                        print(f"                     amp={rejection['amp']:.3f}")
                    if 'rise_len' in rejection:
                        print(f"                     rise_len={rejection['rise_len']} > {rejection['rise_max']}")
                    if 'decay_slope' in rejection:
                        print(f"                     decay_slope={rejection['decay_slope']:.2f} > {rejection['max_allowed']}")
                    if 'snr' in rejection:
                        print(f"                     snr={rejection['snr']:.2f} < {rejection['min_required']}")
    
    return events, mask, debug_info





def compute_qc_metrics_debug(data: Dict[str, Any], cfg: Dict[str, Any], 
                            debug_rois: list = None) -> None:
    """Debug version of QC metrics computation"""
    print("\n=== COMPUTING QC METRICS (DEBUG) ===")
    
    # Get Ca transient parameters with explicit defaults
    ca_cfg = cfg.get("ca_transients", {})
    
    # Print what we actually got from config
    print("\nActual config parameters:")
    for key, default_val in [
        ('smooth_sigma_s', 0.05), ('k_on', 1.5), ('k_off', 0.5), 
        ('min_amp', 0.05), ('min_series_peak_dff', 0.10)
    ]:
        actual_val = ca_cfg.get(key, default_val)
        print(f"  {key}: {actual_val}")
    
    fs = float(cfg["acq"]["fs"])
    dff_clean = data.get('dFF_clean')
    
    if dff_clean is None:
        print("ERROR: No dF/F data available")
        return
    
    N, T = dff_clean.shape
    
    # Use debug ROI list or default to first few
    if debug_rois is None:
        debug_rois = [2, 18, 21]  # ROIs we've been looking at
    
    print(f"\nDebugging ROIs: {debug_rois}")
    
    # Debug each ROI individually
    all_events = []
    all_masks = []
    
    for roi in debug_rois:
        if roi >= N:
            print(f"ROI {roi} out of range (max: {N-1})")
            continue
            
        print(f"\n{'='*50}")
        print(f"DEBUGGING ROI {roi}")
        print(f"{'='*50}")
        
        events, mask, debug_info = detect_ca_transients_debug(
            dff_clean[roi], fs, roi_id=roi, debug=True, **ca_cfg
        )
        
        all_events.append(events)
        all_masks.append(mask)
    
    # Store results for the debugged ROIs
    debug_results = {
        'debug_rois': debug_rois,
        'events': all_events,
        'masks': all_masks
    }
    
    data['debug_ca_results'] = debug_results
    print(f"\n=== DEBUG COMPLETE ===")
    print(f"Processed {len(debug_rois)} ROIs")
    print(f"Total events found: {sum(len(events) for events in all_events)}")



def debug_ca_detection_timing_issue(data: Dict[str, Any], cfg: Dict[str, Any], roi: int) -> None:
    """Debug the timing alignment issue in Ca detection"""
    print(f"\n=== DEBUGGING TIMING ISSUE - ROI {roi} ===")
    
    # Get the segment we're analyzing (first segment 600-640s)
    rv = cfg['review']
    segments = list(rv['detail_segments'])
    fs = float(cfg['acq']['fs'])
    
    segment = segments[0]  # (600, 640)
    sa, sb = segment
    
    print(f"Target segment: {sa}-{sb}s")
    
    # Calculate the slice for this segment
    i0 = int(round(sa * fs))
    i1 = int(round(sb * fs))
    slc = slice(i0, i1)
    
    print(f"Calculated slice: [{i0}:{i1}] (length: {i1-i0})")
    print(f"At {fs} Hz: {(i1-i0)/fs:.1f}s duration")
    
    # Extract actual segment data
    dff_segment = data['dFF_clean'][roi, slc]
    t_segment = np.arange(len(dff_segment)) / fs  # Relative time
    t_absolute = sa + t_segment  # Absolute time
    
    print(f"Segment data shape: {dff_segment.shape}")
    print(f"Time arrays:")
    print(f"  Relative: [{t_segment[0]:.1f}, {t_segment[-1]:.1f}]s")
    print(f"  Absolute: [{t_absolute[0]:.1f}, {t_absolute[-1]:.1f}]s")
    print(f"  dF/F range: [{dff_segment.min():.3f}, {dff_segment.max():.3f}]")
    
    # Run detection on this exact segment
    ca_cfg = cfg.get("ca_transients", {})
    print(f"\nRunning detection on segment...")
    
    events, mask, debug_info = detect_ca_transients_debug_enhanced(
        dff_segment, fs, roi_id=f"{roi}_timing_debug", debug=True, **ca_cfg
    )
    
    print(f"\nDetection results:")
    print(f"  Events found: {len(events)}")
    print(f"  Mask true regions: {np.sum(mask)}")
    
    if len(events) > 0:
        print(f"  Event details (RELATIVE times):")
        for i, e in enumerate(events):
            print(f"    Event {i}: t_on={e['t_on']:.2f}, t_peak={e['t_peak']:.2f}, t_off={e['t_off']:.2f}, amp={e['amp']:.3f}")
        
        print(f"  Event details (ABSOLUTE times - what should be plotted):")
        for i, e in enumerate(events):
            print(f"    Event {i}: t_on={sa + e['t_on']:.2f}, t_peak={sa + e['t_peak']:.2f}, t_off={sa + e['t_off']:.2f}")
    
    # Check if there are obvious transients that should be detected
    print(f"\nVisual inspection of dF/F peaks:")
    peak_threshold = np.percentile(dff_segment, 90)  # Top 10% values
    peak_indices = np.where(dff_segment > peak_threshold)[0]
    
    if len(peak_indices) > 0:
        print(f"  High dF/F points (>{peak_threshold:.3f}):")
        for idx in peak_indices[:10]:  # Show first 10
            abs_time = sa + idx/fs
            print(f"    t={abs_time:.1f}s: dF/F={dff_segment[idx]:.3f}")
        
        if len(peak_indices) > 10:
            print(f"    ... and {len(peak_indices)-10} more")
    
    # Check thresholds
    if 'th_on' in debug_info:
        th_on = debug_info['th_on']
        th_off = debug_info['th_off']
        baseline = debug_info['baseline']
        
        above_on = np.sum(dff_segment > th_on)
        print(f"\nThreshold analysis:")
        print(f"  Baseline: {baseline:.3f}")
        print(f"  Onset threshold: {th_on:.3f}")
        print(f"  Offset threshold: {th_off:.3f}")
        print(f"  Points above onset: {above_on}/{len(dff_segment)} ({100*above_on/len(dff_segment):.1f}%)")
        
        if above_on == 0:
            print(f"  *** NO POINTS ABOVE ONSET THRESHOLD - This explains why no events detected! ***")
            print(f"  Max dF/F in segment: {dff_segment.max():.3f}")
            print(f"  Need to lower thresholds or check noise estimation")



def load_sid_imaging_preprocess(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load SID imaging preprocessing data for timing alignment"""
    print("\n=== LOADING SID IMAGING PREPROCESS DATA ===")
    
    
    
    output_dir = cfg["paths"]["output_dir"]
    pkl_path = os.path.join(output_dir, "sid_imaging_preprocess_data.pkl")
    
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"SID imaging preprocess file not found: {pkl_path}")
    
    print(f"Loading from: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        sid_data = pickle.load(f)
    
    print(f"SID imaging data loaded successfully")
    print(f"Available keys: {list(sid_data.keys())}")
    
    # Check for required timing keys
    required_keys = ['imaging_time',
                     'imaging_fs', 
                     'imaging_duration', 
                     'imaging_duration_session', 
                     'vol_time',
                     'voltage_fs',
                     'vol_duration', 
                     'vol_duration_session',
                     'vol_start',
                     'vol_stim_vis']
    
    found_keys = []
    missing_keys = []
    
    for key in required_keys:
        if key in sid_data:
            found_keys.append(key)
            if 'time' in key or 'start' in key or 'stim_vis' in key:
                print(f"  {key}: shape {sid_data[key].shape}")
            else:
                print(f"  {key}: {sid_data[key]:.1f}s")
        else:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"WARNING: Missing keys: {missing_keys}")
    
    img_data = {}
    for key in found_keys:
        img_data[key] = sid_data[key]
    
    return img_data






def load_sid_behavioral_preprocess(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load SID behavioral preprocessing data for timing and trial information"""
    print("\n=== LOADING SID BEHAVIORAL PREPROCESS DATA ===")
    
    output_dir = cfg["paths"]["output_dir"]
    pkl_path = os.path.join(output_dir, "sid_behavioral_preprocess_data.pkl")
    
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"SID behavioral preprocess file not found: {pkl_path}")
    
    print(f"Loading from: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        session_data = pickle.load(f)
    
    print(f"SID behavioral data loaded successfully")
    print(f"Available keys: {list(session_data.keys())}")
    
    # Extract required keys
    required_keys = [
        'df_trials',
        'session_id', 
        'subject_name'
    ]
    
    # Session info keys that should be moved to top level
    session_info_keys = [
        'unique_isis',
        'mean_isi',
        'long_isis', 
        'short_isis'
    ]
    
    behavioral_data = {}
    found_keys = []
    missing_keys = []
    
    # Load direct keys
    for key in required_keys:
        if key in session_data:
            behavioral_data[key] = session_data[key]
            found_keys.append(key)
            if key == 'df_trials':
                print(f"  {key}: shape {session_data[key].shape}")
            else:
                print(f"  {key}: {session_data[key]}")
        else:
            missing_keys.append(key)
    
    # Load session_info keys to top level
    session_info = session_data.get('session_info', {})
    for key in session_info_keys:
        if key in session_info:
            behavioral_data[key] = session_info[key]
            found_keys.append(f'session_info.{key}')
            print(f"  {key}: {session_info[key]}")
        else:
            missing_keys.append(f'session_info.{key}')
    
    if missing_keys:
        print(f"WARNING: Missing keys: {missing_keys}")
    
    print(f"Extracted {len(found_keys)} required keys")
    
    return behavioral_data





























def align_trial_timestamps_to_vol_start(data: Dict[str, Any], cfg: Dict[str, Any], 
                                       tolerance_s: float = 0.05) -> Dict[str, Any]:
    """
    Align trial_start_timestamp to vol_start rising edges to correct for timing drift
    
    Parameters:
    - tolerance_s: Maximum allowed deviation before correction (default 0.05s = 50ms)
    """
    print("\n=== ALIGNING TRIAL TIMESTAMPS TO VOL_START RISING EDGES ===")
    
    # Get required data
    df_trials = data.get('df_trials')
    vol_start = data.get('vol_start')
    vol_time = data.get('vol_time')
    
    if df_trials is None or vol_start is None or vol_time is None:
        raise ValueError("Missing required data: df_trials, vol_start, or vol_time")
    
    print(f"Input data:")
    print(f"  df_trials: {len(df_trials)} trials")
    print(f"  vol_start: {len(vol_start)} samples")
    print(f"  vol_time: {len(vol_time)} samples, range: {vol_time[0]:.1f} to {vol_time[-1]:.1f}s")
    print(f"  Tolerance: ±{tolerance_s*1000:.0f}ms")
    
    # Detect vol_start rising edges
    print(f"\nDetecting vol_start rising edges...")
    
    # Threshold vol_start signal (assume digital-like signal)
    vol_threshold = (np.max(vol_start) + np.min(vol_start)) / 2
    vol_binary = vol_start > vol_threshold
    
    # Find rising edges (0 -> 1 transitions)
    rising_edges = np.where(np.diff(vol_binary.astype(int)) > 0)[0] + 1  # +1 because diff shifts index
    rising_edge_times = vol_time[rising_edges]
    
    print(f"  vol_start threshold: {vol_threshold:.3f}")
    print(f"  Found {len(rising_edges)} rising edges")
    print(f"  Rising edge times range: {rising_edge_times[0]:.1f} to {rising_edge_times[-1]:.1f}s")
    print(f"  Sample rising edges: {rising_edge_times[:5].round(1)} (first 5)")
    
    # Check alignment for each trial
    print(f"\n=== TRIAL ALIGNMENT CHECK ===")
    
    n_trials = len(df_trials)
    adjustments_made = 0
    total_drift_ms = 0
    max_drift_ms = 0
    drift_rms_ms = 0
    
    # Store original timestamps for comparison
    original_timestamps = df_trials['trial_start_timestamp'].copy()
    adjusted_timestamps = df_trials['trial_start_timestamp'].copy()
    
    alignment_results = []
    
    for i in range(n_trials):
        trial_time = df_trials.iloc[i]['trial_start_timestamp']
        
        # Find closest vol_start rising edge
        time_diffs = np.abs(rising_edge_times - trial_time)
        closest_idx = np.argmin(time_diffs)
        closest_edge_time = rising_edge_times[closest_idx]
        drift_s = trial_time - closest_edge_time
        drift_ms = drift_s * 1000
        
        # Check if adjustment needed
        needs_adjustment = abs(drift_s) > tolerance_s
        
        if needs_adjustment:
            adjusted_timestamps.iloc[i] = closest_edge_time
            adjustments_made += 1
            adjustment_str = f"ADJUSTED: {drift_ms:+.1f}ms -> edge"
        else:
            adjustment_str = f"OK: {drift_ms:+.1f}ms"
        
        # Store results
        alignment_results.append({
            'trial': i,
            'original_time': trial_time,
            'closest_edge_time': closest_edge_time,
            'drift_ms': drift_ms,
            'needs_adjustment': needs_adjustment,
            'final_time': adjusted_timestamps.iloc[i]
        })
        
        # Statistics
        total_drift_ms += abs(drift_ms)
        max_drift_ms = max(max_drift_ms, abs(drift_ms))
        drift_rms_ms += drift_ms ** 2
        
        # Print every 10th trial or if adjustment made
        if i % 10 == 0 or needs_adjustment or i < 5 or i >= n_trials - 5:
            print(f"  Trial {i:3d}: {trial_time:7.1f}s -> edge {closest_edge_time:7.1f}s | {adjustment_str}")
    
    # Compute RMS drift
    drift_rms_ms = np.sqrt(drift_rms_ms / n_trials)
    mean_abs_drift_ms = total_drift_ms / n_trials
    
    # Update the DataFrame
    data['df_trials'] = df_trials.copy()
    data['df_trials']['trial_start_timestamp'] = adjusted_timestamps
    
    # Store alignment information
    data['trial_alignment_info'] = {
        'vol_start_rising_edges': rising_edge_times,
        'vol_start_rising_edge_indices': rising_edges,
        'alignment_results': alignment_results,
        'original_timestamps': original_timestamps,
        'adjusted_timestamps': adjusted_timestamps,
        'tolerance_s': tolerance_s,
        'adjustments_made': adjustments_made,
        'stats': {
            'mean_abs_drift_ms': mean_abs_drift_ms,
            'max_drift_ms': max_drift_ms,
            'rms_drift_ms': drift_rms_ms
        }
    }
    
    # Summary
    print(f"\n=== ALIGNMENT SUMMARY ===")
    print(f"  Trials processed: {n_trials}")
    print(f"  Adjustments made: {adjustments_made} ({100*adjustments_made/n_trials:.1f}%)")
    print(f"  Mean absolute drift: {mean_abs_drift_ms:.1f}ms")
    print(f"  RMS drift: {drift_rms_ms:.1f}ms")
    print(f"  Maximum drift: {max_drift_ms:.1f}ms")
    
    if adjustments_made > 0:
        print(f"  *** {adjustments_made} trials had timing drift > {tolerance_s*1000:.0f}ms and were corrected ***")
        
        # Show worst cases
        worst_drifts = sorted(alignment_results, key=lambda x: abs(x['drift_ms']), reverse=True)[:3]
        print(f"  Worst drift cases:")
        for result in worst_drifts:
            if result['needs_adjustment']:
                print(f"    Trial {result['trial']}: {result['drift_ms']:+.1f}ms drift corrected")
    else:
        print(f"  *** All trials were within tolerance - no corrections needed ***")
    
    # Timing validation
    print(f"\n=== POST-CORRECTION VALIDATION ===")
    final_drifts = []
    for result in alignment_results:
        final_time = result['final_time']
        edge_time = result['closest_edge_time']
        final_drift_ms = (final_time - edge_time) * 1000
        final_drifts.append(abs(final_drift_ms))
    
    max_final_drift_ms = max(final_drifts)
    mean_final_drift_ms = np.mean(final_drifts)
    
    print(f"  Post-correction mean drift: {mean_final_drift_ms:.1f}ms")
    print(f"  Post-correction max drift: {max_final_drift_ms:.1f}ms")
    
    if max_final_drift_ms > tolerance_s * 1000:
        print(f"  *** WARNING: Some trials still exceed tolerance after correction ***")
    else:
        print(f"  ✓ All trials now within tolerance")
    
    return data

def plot_trial_alignment_diagnostic(data: Dict[str, Any], cfg: Dict[str, Any], 
                                   show_n_trials: int = 50, save: bool = True) -> Any:
    """Plot diagnostic information about trial timestamp alignment"""
    print("\n=== PLOTTING TRIAL ALIGNMENT DIAGNOSTICS ===")
    
    if 'trial_alignment_info' not in data:
        print("No trial alignment info found - run align_trial_timestamps_to_vol_start first")
        return None
    
    align_info = data['trial_alignment_info']
    alignment_results = align_info['alignment_results']
    vol_edges = align_info['vol_start_rising_edges']
    
    # Limit to requested number of trials for readability
    n_trials = min(show_n_trials, len(alignment_results))
    results_subset = alignment_results[:n_trials]
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # 1. Drift over time
    ax = axes[0]
    
    trial_nums = [r['trial'] for r in results_subset]
    drift_ms = [r['drift_ms'] for r in results_subset]
    adjusted = [r['needs_adjustment'] for r in results_subset]
    
    # Color code by whether adjustment was made
    colors = ['red' if adj else 'blue' for adj in adjusted]
    
    scatter = ax.scatter(trial_nums, drift_ms, c=colors, alpha=0.7, s=30)
    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax.axhline(align_info['tolerance_s']*1000, color='orange', linestyle='--', alpha=0.7, label=f'Tolerance (±{align_info["tolerance_s"]*1000:.0f}ms)')
    ax.axhline(-align_info['tolerance_s']*1000, color='orange', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Timing Drift (ms)')
    ax.set_title(f'Trial Timing Drift from vol_start Rising Edges (first {n_trials} trials)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add custom legend for colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Within tolerance'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Adjusted')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 2. vol_start signal with detected edges
    ax = axes[1]
    
    # Show a representative section of vol_start
    vol_start = data['vol_start']
    vol_time = data['vol_time']
    
    # Find a section that includes several trials
    if len(results_subset) > 0:
        start_time = results_subset[0]['original_time'] - 10
        end_time = results_subset[min(5, len(results_subset)-1)]['original_time'] + 10
        
        time_mask = (vol_time >= start_time) & (vol_time <= end_time)
        vol_section = vol_start[time_mask]
        time_section = vol_time[time_mask]
        
        ax.plot(time_section, vol_section, 'k-', linewidth=0.8, alpha=0.7, label='vol_start')
        
        # Mark rising edges in this section
        edges_in_section = vol_edges[(vol_edges >= start_time) & (vol_edges <= end_time)]
        if len(edges_in_section) > 0:
            # Find closest indices in the section
            edge_y_values = []
            for edge_time in edges_in_section:
                closest_idx = np.argmin(np.abs(time_section - edge_time))
                edge_y_values.append(vol_section[closest_idx])
            
            ax.scatter(edges_in_section, edge_y_values, c='red', s=50, marker='^', 
                      zorder=10, label='Rising edges')
        
        # Mark original trial times
        for r in results_subset[:6]:  # Show first few trials
            if start_time <= r['original_time'] <= end_time:
                ax.axvline(r['original_time'], color='blue', linestyle='--', alpha=0.6)
                ax.text(r['original_time'], ax.get_ylim()[1]*0.9, f"T{r['trial']}", 
                       rotation=90, ha='right', va='top', fontsize=8)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('vol_start')
        ax.set_title('vol_start Signal with Detected Rising Edges and Trial Times')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Histogram of drift magnitudes
    ax = axes[2]
    
    all_drift_abs = [abs(r['drift_ms']) for r in alignment_results]
    adjusted_drift_abs = [abs(r['drift_ms']) for r in alignment_results if r['needs_adjustment']]
    
    ax.hist(all_drift_abs, bins=30, alpha=0.7, color='blue', label=f'All trials (n={len(all_drift_abs)})')
    if len(adjusted_drift_abs) > 0:
        ax.hist(adjusted_drift_abs, bins=30, alpha=0.7, color='red', label=f'Adjusted (n={len(adjusted_drift_abs)})')
    
    ax.axvline(align_info['tolerance_s']*1000, color='orange', linestyle='--', 
              label=f'Tolerance ({align_info["tolerance_s"]*1000:.0f}ms)')
    
    ax.set_xlabel('|Drift| (ms)')
    ax.set_ylabel('Number of Trials')
    ax.set_title('Distribution of Timing Drift Magnitudes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        save_dir = cfg['overlay']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'trial_alignment_diagnostics.png')
        fig.savefig(save_path, dpi=cfg['overlay']['dpi'], bbox_inches='tight')
        print(f"Trial alignment diagnostics saved: {save_path}")
    
    plt.show()
    return fig















def plot_single_trial_with_context(data: Dict[str, Any], cfg: Dict[str, Any], 
                                  trial_idx: int = 0,
                                  roi: int = 5,
                                  pre_trial_sec: float = 5.0,
                                  post_trial_sec: float = 10.0,
                                  save: bool = False, save_dir: str = None) -> Any:
    """Plot a single trial with surrounding context - much more readable"""
    print(f"\n=== PLOTTING TRIAL {trial_idx}, ROI {roi} ===")
    
    # Get data
    df_trials = data['df_trials']
    fs = cfg['acq']['fs']
    
    if trial_idx >= len(df_trials):
        raise ValueError(f"Trial {trial_idx} not found (only {len(df_trials)} trials)")
    
    trial = df_trials.iloc[trial_idx]
    trial_start_time = trial['trial_start_timestamp']  # Already in seconds
    
    # Define time window
    window_start = trial_start_time - pre_trial_sec
    window_end = trial_start_time + post_trial_sec
    
    # Convert to frame indices
    start_frame = int(window_start * fs)
    end_frame = int(window_end * fs)
    
    # Ensure we're within bounds
    start_frame = max(0, start_frame)
    end_frame = min(len(data['dFF_clean'][roi]), end_frame)
    
    # Extract data for this window
    frame_times = np.arange(start_frame, end_frame) / fs
    dff_window = data['dFF_clean'][roi][start_frame:end_frame]
    
    # Check if we have spike data
    has_spikes = 'spks_oasis' in data and data['spks_oasis'].shape[0] > roi
    if has_spikes:
        spikes_window = data['spks_oasis'][roi][start_frame:end_frame]
    
    print(f"  Trial start: {trial_start_time:.1f}s")
    print(f"  Window: {window_start:.1f} to {window_end:.1f}s")
    print(f"  Frames: {start_frame} to {end_frame}")
    print(f"  Data points: {len(frame_times)}")
    print(f"  Has spikes: {has_spikes}")
    
    # Create figure
    n_subplots = 2 if has_spikes else 1
    fig, axes = plt.subplots(n_subplots, 1, figsize=(15, 8), sharex=True)
    if n_subplots == 1:
        axes = [axes]
    
    # 1. dF/F trace
    ax = axes[0]
    ax.plot(frame_times, dff_window, 'k-', linewidth=1.0, alpha=0.8)
    ax.set_ylabel('dF/F')
    ax.set_title(f'Trial {trial_idx} - ROI {roi} (ISI: {trial["isi"]:.0f}ms)')
    
    # Add trial event markers
    add_single_trial_markers(ax, trial, trial_start_time)
    
    # Add ISI shading for this trial
    add_single_trial_isi_shading(ax, trial, trial_start_time)
    
    # 2. Spike data if available
    if has_spikes:
        ax = axes[1]
        ax.plot(frame_times, spikes_window, 'r-', linewidth=1.0, alpha=0.8)
        ax.set_ylabel('Spikes')
        
        # Mark significant spike events
        spike_threshold = 0.1 * np.max(spikes_window) if np.max(spikes_window) > 0 else 0
        if spike_threshold > 0:
            spike_events = spikes_window > spike_threshold
            if np.any(spike_events):
                ax.scatter(frame_times[spike_events], spikes_window[spike_events], 
                          c='red', s=20, alpha=0.8, zorder=5)
        
        # Add trial markers to spike plot too
        add_single_trial_markers(ax, trial, trial_start_time)
    
    # Format plot
    axes[-1].set_xlabel('Time (s)')
    
    # Add trial info text
    trial_info = f"ISI: {trial['isi']:.0f}ms"
    if 'mouse_correct' in trial:
        trial_info += f" | Correct: {trial['mouse_correct']}"
    if 'rewarded' in trial:
        trial_info += f" | Rewarded: {trial['rewarded']}"
    
    fig.suptitle(f"Single Trial View - {trial_info}", fontsize=14)
    
    plt.tight_layout()
    
    if save and save_dir:
        save_path = os.path.join(save_dir, f'trial_{trial_idx:03d}_roi_{roi:04d}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Single trial figure saved: {save_path}")
    
    return fig

def add_single_trial_markers(ax, trial, trial_start_time):
    """Add event markers for a single trial - FIXED: only ISI is in ms"""
    # Trial start
    ax.axvline(trial_start_time, color='black', linestyle='-', alpha=0.8, linewidth=2, label='Trial start')
    
    # Flash events - these are already in seconds, don't divide by 1000
    if 'start_flash_1' in trial:
        flash1_start = trial_start_time + trial['start_flash_1']  # Already in seconds
        ax.axvline(flash1_start, color='blue', linestyle='-', alpha=0.7, linewidth=1, label='Flash 1 start')
    
    if 'end_flash_1' in trial:
        flash1_end = trial_start_time + trial['end_flash_1']  # Already in seconds
        ax.axvline(flash1_end, color='blue', linestyle='--', alpha=0.7, linewidth=1, label='Flash 1 end')
    
    if 'start_flash_2' in trial:
        flash2_start = trial_start_time + trial['start_flash_2']  # Already in seconds
        ax.axvline(flash2_start, color='purple', linestyle='-', alpha=0.7, linewidth=1, label='Flash 2 start')
    
    # Choice period - already in seconds
    if 'choice_start' in trial:
        choice_start = trial_start_time + trial['choice_start']  # Already in seconds
        ax.axvline(choice_start, color='orange', linestyle='-', alpha=0.7, linewidth=1, label='Choice start')
    
    # Lick/response - already in seconds
    if 'lick_start' in trial and not pd.isna(trial['lick_start']):
        lick_time = trial_start_time + trial['lick_start']  # Already in seconds
        ax.axvline(lick_time, color='green', linestyle='-', alpha=0.8, linewidth=2, label='Lick')
    
    ax.legend(loc='upper right', fontsize=8)

def add_single_trial_isi_shading(ax, trial, trial_start_time):
    """Add ISI shading for a single trial - FIXED: only ISI duration is in ms"""
    if 'end_flash_1' in trial and 'start_flash_2' in trial:
        isi_start = trial_start_time + trial['end_flash_1']  # Already in seconds
        isi_end = trial_start_time + trial['start_flash_2']    # Already in seconds
        
        # Color by ISI duration - this is in ms, so display as ms
        isi_duration_ms = trial['isi']  # This is in ms
        if isi_duration_ms > 1000.0:  # Long ISI (>1 second)
            color = 'lightblue'
            label = f'ISI ({isi_duration_ms:.0f}ms - Long)'
        else:  # Short ISI (≤1 second)
            color = 'lightcoral'
            label = f'ISI ({isi_duration_ms:.0f}ms - Short)'
        
        ax.axvspan(isi_start, isi_end, alpha=0.3, color=color, zorder=1, label=label)

def add_trial_markers_to_axis(ax, df_trials, trial_alignment, fs):
    """Add trial event markers to an axis - FIXED: only ISI is in ms"""
    
    # Get the event column name for alignment
    if trial_alignment == 'trial_start':
        event_offset = 0  # trial_start is always 0 in relative time
    elif trial_alignment == 'start_flash_1':
        event_offset = 'start_flash_1'
    elif trial_alignment == 'end_flash_1':
        event_offset = 'end_flash_1'
    else:
        event_offset = 0  # fallback to trial_start
    
    # Add vertical lines for trial events
    for _, trial in df_trials.iterrows():
        # trial_start_timestamp is already in seconds
        trial_abs_time = trial['trial_start_timestamp']  # Already in seconds
        
        # Add offset for the specific event we're aligning to
        if isinstance(event_offset, str):
            event_time = trial_abs_time + trial[event_offset]  # Already in seconds, no conversion needed
        else:
            event_time = trial_abs_time  # trial_start case
        
        # Color code by trial type or outcome
        if 'mouse_correct' in trial:
            if trial['mouse_correct']:
                color = 'green'
                alpha = 0.6
            else:
                color = 'red'
                alpha = 0.6
        elif 'rewarded' in trial:
            if trial['rewarded']:
                color = 'green'
                alpha = 0.6
            elif trial['punished']:
                color = 'red'
                alpha = 0.6
            else:
                color = 'orange'  # did not choose
                alpha = 0.4
        else:
            color = 'blue'
            alpha = 0.4
        
        ax.axvline(event_time, color=color, linestyle='-', alpha=alpha, linewidth=1)

def add_isi_shading_to_axis(ax, df_trials, fs):
    """Add ISI period shading to an axis - FIXED: only ISI duration is in ms"""
    for _, trial in df_trials.iterrows():
        # trial_start_timestamp is already in seconds
        trial_abs_time = trial['trial_start_timestamp']  # Already in seconds
        
        if 'end_flash_1' in trial and 'start_flash_2' in trial:
            # These times are already in seconds, no conversion needed
            isi_start = trial_abs_time + trial['end_flash_1']  # Already in seconds
            isi_end = trial_abs_time + trial['start_flash_2']    # Already in seconds
            
            # Color code by ISI duration (this value is in ms)
            isi_duration_ms = trial['isi']  # This is in ms
            if isi_duration_ms > 1000.0:  # Long ISI (>1 second)
                color = 'lightblue'
            else:  # Short ISI (≤1 second)
                color = 'lightcoral'
            
            ax.axvspan(isi_start, isi_end, alpha=0.2, color=color, zorder=1)

def batch_plot_single_trials(data: Dict[str, Any], cfg: Dict[str, Any],
                            trial_indices: list = None,
                            rois: list = None,
                            pre_trial_sec: float = 5.0,
                            post_trial_sec: float = 10.0) -> None:
    """Batch plot multiple single trial views"""
    print(f"\n=== BATCH PLOTTING SINGLE TRIALS ===")
    
    if trial_indices is None:
        # Plot first 10 trials by default
        trial_indices = list(range(min(10, len(data['df_trials']))))
    
    if rois is None:
        # Plot a few interesting ROIs
        rois = [2, 5, 18]  # Your previously interesting ROIs
    
    # Setup save directory
    rv = cfg['review']
    save_dir = os.path.join(rv['save_dir'], 'single_trials')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Plotting {len(trial_indices)} trials × {len(rois)} ROIs...")
    
    for trial_idx in trial_indices:
        for roi in rois:
            if trial_idx % 5 == 0 and roi == rois[0]:
                print(f"  Progress: Trial {trial_idx}/{max(trial_indices)}")
            
            fig = plot_single_trial_with_context(
                data, cfg, trial_idx, roi,
                pre_trial_sec=pre_trial_sec,
                post_trial_sec=post_trial_sec,
                save=True, save_dir=save_dir
            )
            plt.close(fig)
    
    print(f"Batch single trial plotting complete: {len(trial_indices)}×{len(rois)} figures saved to {save_dir}")


































def save_sid_imaging_data(data: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """Save complete SID imaging data as pickle file"""
    print("\n=== SAVING SID IMAGING DATA ===")
    
    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, "sid_imaging_data.pkl")
    
    # Create comprehensive save structure
    save_data = {
        # === IMAGING DATA ===
        'dFF_clean': data['dFF_clean'],
        'dFF': data['dFF'],
        'F': data['F'],
        'Fc': data['Fc'],
        'F0': data['F0'],
        'Fneu': data.get('Fneu'),
        
        # === SPIKE DATA ===
        'spks_oasis': data.get('spks_oasis'),
        
        # === ROI FEATURES & CLASSIFICATION ===
        'roi_features': data['roi_features'],
        'roi_labels': data['roi_labels'],
        'stat': data['stat'],  # Original Suite2p stat structure
        'anatomy_image': data['anatomy_image'],
        
        # === ROI INDEX MAPPING (CRITICAL) ===
        'roi_index_map': data['roi_index_map'],  # filtered <-> original Suite2p mapping
        'roi_mask': data.get('roi_mask'),  # original filtering mask
        
        # === IMAGING METADATA ===
        'imaging_metadata': data['imaging_metadata'],  # pixel size, etc.
        
        # === TIMING & BEHAVIORAL DATA ===
        'df_trials': data['df_trials'],
        'trial_alignment_info': data.get('trial_alignment_info'),
        'vol_time': data['vol_time'],
        'vol_start': data['vol_start'],
        'vol_stim_vis': data.get('vol_stim_vis'),
        'imaging_time': data['imaging_time'],
        'imaging_fs': data['imaging_fs'],
        
        # === QC METRICS ===
        'qc_metrics': data.get('qc_metrics'),
        'neuropil_a': data.get('neuropil_a'),
        
        # === SESSION INFO ===
        'session_id': data.get('session_id'),
        'subject_name': data.get('subject_name'),
        'unique_isis': data.get('unique_isis'),
        'short_isis': data.get('short_isis'),
        'long_isis': data.get('long_isis'),
        
        # === PROCESSING METADATA ===
        'processing_timestamp': pd.Timestamp.now().isoformat(),
        'config_used': cfg.copy(),
    }
    
    # Verify ROI index mapping exists
    if 'roi_index_map' not in save_data or save_data['roi_index_map'] is None:
        print("WARNING: ROI index mapping missing - creating now...")
        save_data['roi_index_map'] = create_roi_index_mapping(data)['roi_index_map']
    
    # Print summary of what we're saving
    print(f"Saving comprehensive SID imaging data:")
    print(f"  Output file: {save_path}")
    print(f"  ROIs: {save_data['F'].shape[0]} (filtered from {save_data['roi_index_map']['n_original']} original)")
    print(f"  Timepoints: {save_data['F'].shape[1]} ({save_data['F'].shape[1]/save_data['imaging_fs']:.1f}s)")
    print(f"  Trials: {len(save_data['df_trials'])}")
    print(f"  Pixel size: {save_data['imaging_metadata']['pixel_size_um']:.3f} µm/pixel")
    
    # ROI classification summary
    if save_data['roi_labels']:
        from collections import Counter
        label_counts = Counter(save_data['roi_labels'])
        print(f"  Classification: {dict(label_counts)}")
    
    # Show ROI index mapping info
    roi_map = save_data['roi_index_map']
    print(f"  Index mapping: {roi_map['n_filtered']} filtered -> {roi_map['n_original']} original Suite2p")
    print(f"    Example: filtered ROI 0 -> original ROI {roi_map['filtered_to_original'][0]}")
    if roi_map['n_filtered'] > 5:
        print(f"    Example: filtered ROI 5 -> original ROI {roi_map['filtered_to_original'][5]}")
    
    # Save the data
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Verify save and get file size
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  File saved successfully: {file_size_mb:.1f} MB")
    
    # Test loading to verify integrity
    try:
        with open(save_path, 'rb') as f:
            test_load = pickle.load(f)
        print(f"  File integrity verified ✓")
        
        # Quick verification of key arrays
        print(f"  Verification:")
        print(f"    dFF_clean shape: {test_load['dFF_clean'].shape}")
        print(f"    ROI mapping keys: {len(test_load['roi_index_map']['filtered_to_original'])}")
        print(f"    Trial count: {len(test_load['df_trials'])}")
        
    except Exception as e:
        print(f"  WARNING: File verification failed: {e}")
    
    return save_path

def load_sid_imaging_data(file_path: str) -> Dict[str, Any]:
    """Load SID imaging data from pickle file"""
    print(f"\n=== LOADING SID IMAGING DATA ===")
    print(f"Loading from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SID imaging data file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"SID imaging data loaded successfully")
    print(f"  ROIs: {data['F'].shape[0]}")
    print(f"  Timepoints: {data['F'].shape[1]} ({data['F'].shape[1]/data['imaging_fs']:.1f}s)")
    print(f"  Trials: {len(data['df_trials'])}")
    
    if 'roi_index_map' in data:
        roi_map = data['roi_index_map']
        print(f"  ROI mapping: {roi_map['n_filtered']} filtered from {roi_map['n_original']} original")
    
    if 'processing_timestamp' in data:
        print(f"  Processed: {data['processing_timestamp']}")
    
    return data

def get_original_suite2p_roi_index(data: Dict[str, Any], filtered_roi_idx: int) -> int:
    """Helper function to get original Suite2p ROI index from filtered index"""
    if 'roi_index_map' not in data:
        raise ValueError("ROI index mapping not found in data")
    
    roi_map = data['roi_index_map']
    if filtered_roi_idx not in roi_map['filtered_to_original']:
        raise ValueError(f"Filtered ROI index {filtered_roi_idx} not found in mapping")
    
    return roi_map['filtered_to_original'][filtered_roi_idx]

def get_filtered_roi_index(data: Dict[str, Any], original_suite2p_idx: int) -> Optional[int]:
    """Helper function to get filtered ROI index from original Suite2p index"""
    if 'roi_index_map' not in data:
        raise ValueError("ROI index mapping not found in data")
    
    roi_map = data['roi_index_map']
    return roi_map['original_to_filtered'].get(original_suite2p_idx)





























def diagnose_problematic_roi_signals(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """Comprehensive diagnosis of problematic ROI signals across all processing stages"""
    print("\n=== COMPREHENSIVE ROI SIGNAL DIAGNOSIS ===")
    
    # First, identify the worst cases from our previous diagnosis
    dff_clean = data['dFF_clean']
    n_rois = dff_clean.shape[0]
    
    exploded_rois = []
    for roi_idx in range(n_rois):
        roi_trace = dff_clean[roi_idx, :]
        roi_min = np.min(roi_trace)
        roi_max = np.max(roi_trace)
        roi_range = roi_max - roi_min
        
        if roi_max > 50 or roi_min < -50:
            exploded_rois.append({
                'roi': roi_idx,
                'min': roi_min,
                'max': roi_max,
                'range': roi_range
            })
    
    # Sort by severity and pick the worst cases
    exploded_sorted = sorted(exploded_rois, key=lambda x: x['range'], reverse=True)
    worst_rois = [r['roi'] for r in exploded_sorted[:5]]  # Top 5 worst
    
    print(f"Analyzing {len(worst_rois)} most problematic ROIs: {worst_rois}")
    
    # Also pick some "normal" ROIs for comparison
    all_ranges = []
    for roi_idx in range(n_rois):
        roi_trace = dff_clean[roi_idx, :]
        roi_range = np.max(roi_trace) - np.min(roi_trace)
        all_ranges.append((roi_idx, roi_range))
    
    # Find ROIs with reasonable dF/F ranges (not too small, not too big)
    reasonable_ranges = [(roi, r) for roi, r in all_ranges if 0.5 < r < 5.0]
    if len(reasonable_ranges) >= 3:
        reasonable_rois = [roi for roi, _ in reasonable_ranges[:3]]
    else:
        reasonable_rois = [0, 1, 2]  # Fallback
    
    print(f"Comparing with normal ROIs: {reasonable_rois}")
    
    # Create comprehensive diagnostic figure
    all_rois_to_check = worst_rois + reasonable_rois
    n_rois_check = len(all_rois_to_check)
    
    fig, axes = plt.subplots(n_rois_check, 6, figsize=(24, 4*n_rois_check))
    if n_rois_check == 1:
        axes = axes.reshape(1, -1)
    
    for plot_idx, roi in enumerate(all_rois_to_check):
        roi_type = "PROBLEMATIC" if roi in worst_rois else "NORMAL"
        
        print(f"\n--- ROI {roi} ({roi_type}) ---")
        
        # Get all the signals for this ROI
        F_roi = data['F'][roi]
        Fneu_roi = data['Fneu'][roi]
        Fc_roi = data['Fc'][roi]
        F0_roi = data['F0'][roi]
        dFF_roi = data['dFF_clean'][roi]
        alpha_roi = data['neuropil_a'][roi]
        
        # Time vector
        fs = cfg['acq']['fs']
        t = np.arange(len(F_roi)) / fs
        
        # Basic statistics
        print(f"  Alpha coefficient: {alpha_roi:.3f}")
        print(f"  F range: [{F_roi.min():.1f}, {F_roi.max():.1f}], mean: {F_roi.mean():.1f}")
        print(f"  Fneu range: [{Fneu_roi.min():.1f}, {Fneu_roi.max():.1f}], mean: {Fneu_roi.mean():.1f}")
        print(f"  Fc range: [{Fc_roi.min():.1f}, {Fc_roi.max():.1f}], mean: {Fc_roi.mean():.1f}")
        print(f"  F0 range: [{F0_roi.min():.1f}, {F0_roi.max():.1f}], mean: {F0_roi.mean():.1f}")
        print(f"  dF/F range: [{dFF_roi.min():.1f}, {dFF_roi.max():.1f}], mean: {dFF_roi.mean():.1f}")
        
        # Check for problematic patterns
        negative_fc_pct = 100 * np.sum(Fc_roi < 0) / len(Fc_roi)
        near_zero_f0_pct = 100 * np.sum(F0_roi < 0.1) / len(F0_roi)
        negative_f0_pct = 100 * np.sum(F0_roi < 0) / len(F0_roi)
        
        print(f"  Negative Fc: {negative_fc_pct:.1f}%")
        print(f"  Near-zero F0 (<0.1): {near_zero_f0_pct:.1f}%")
        print(f"  Negative F0: {negative_f0_pct:.1f}%")
        
        # 1. Raw F and Fneu
        ax = axes[plot_idx, 0]
        ax.plot(t, F_roi, 'k-', linewidth=0.5, alpha=0.8, label='F')
        ax.plot(t, Fneu_roi, 'm-', linewidth=0.5, alpha=0.8, label='Fneu')
        ax.set_title(f'ROI {roi} ({roi_type})\nRaw F & Fneu\nα={alpha_roi:.3f}')
        ax.set_ylabel('Fluorescence')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 2. F vs alpha*Fneu (to see the correction)
        ax = axes[plot_idx, 1]
        ax.plot(t, F_roi, 'k-', linewidth=0.5, alpha=0.8, label='F')
        ax.plot(t, alpha_roi * Fneu_roi, 'r-', linewidth=0.5, alpha=0.8, label=f'{alpha_roi:.2f}×Fneu')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'Neuropil Correction\nF vs α×Fneu')
        ax.set_ylabel('Fluorescence')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 3. Fc (neuropil corrected)
        ax = axes[plot_idx, 2]
        ax.plot(t, Fc_roi, 'b-', linewidth=0.5, alpha=0.8)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'Fc (Corrected)\nRange: [{Fc_roi.min():.1f}, {Fc_roi.max():.1f}]\n{negative_fc_pct:.1f}% negative')
        ax.set_ylabel('Fc')
        ax.grid(True, alpha=0.3)
        
        # 4. F0 baseline
        ax = axes[plot_idx, 3]
        ax.plot(t, Fc_roi, 'b-', linewidth=0.3, alpha=0.5, label='Fc')
        ax.plot(t, F0_roi, 'orange', linewidth=0.8, alpha=0.8, label='F0')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'Baseline F0\nRange: [{F0_roi.min():.1f}, {F0_roi.max():.1f}]\n{negative_f0_pct:.1f}% negative')
        ax.set_ylabel('Fc & F0')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 5. dF/F calculation breakdown
        ax = axes[plot_idx, 4]
        # Show (Fc - F0) numerator and F0 denominator
        numerator = Fc_roi - F0_roi
        ax.plot(t, numerator, 'g-', linewidth=0.5, alpha=0.8, label='Fc-F0 (num)')
        ax.plot(t, F0_roi, 'orange', linewidth=0.5, alpha=0.8, label='F0 (denom)')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'dF/F Components\nNumerator & Denominator')
        ax.set_ylabel('Fc-F0, F0')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 6. Final dF/F
        ax = axes[plot_idx, 5]
        ax.plot(t, dFF_roi, 'purple', linewidth=0.5, alpha=0.8)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'Final dF/F\nRange: [{dFF_roi.min():.1f}, {dFF_roi.max():.1f}]')
        ax.set_ylabel('dF/F')
        ax.grid(True, alpha=0.3)
        
        # Add session time info
        if plot_idx == n_rois_check - 1:  # Last row
            for col in range(6):
                axes[plot_idx, col].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()
    
    # Session-wide signal comparison
    print(f"\n=== SESSION-WIDE SIGNAL ANALYSIS ===")
    
    # Look at session-wide patterns
    F_session_mean = np.mean(data['F'], axis=0)
    Fneu_session_mean = np.mean(data['Fneu'], axis=0)
    Fc_session_mean = np.mean(data['Fc'], axis=0)
    F0_session_mean = np.mean(data['F0'], axis=0)
    
    # Check for drift/trends
    t_session = np.arange(len(F_session_mean)) / fs
    
    # Linear trend analysis
    F_trend = np.polyfit(t_session, F_session_mean, 1)[0]
    Fneu_trend = np.polyfit(t_session, Fneu_session_mean, 1)[0]
    Fc_trend = np.polyfit(t_session, Fc_session_mean, 1)[0]
    
    print(f"Session-wide trends (slope per second):")
    print(f"  F trend: {F_trend:.3f} units/s")
    print(f"  Fneu trend: {Fneu_trend:.3f} units/s")
    print(f"  Fc trend: {Fc_trend:.3f} units/s")
    
    # Session-wide statistics
    print(f"Session-wide statistics:")
    print(f"  F: mean={F_session_mean.mean():.1f}, std={F_session_mean.std():.1f}")
    print(f"  Fneu: mean={Fneu_session_mean.mean():.1f}, std={Fneu_session_mean.std():.1f}")
    print(f"  Fc: mean={Fc_session_mean.mean():.1f}, std={Fc_session_mean.std():.1f}")
    
    # Plot session-wide signals
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    
    ax = axes[0]
    ax.plot(t_session, F_session_mean, 'k-', linewidth=0.8, alpha=0.8, label='Session F mean')
    ax.plot(t_session, Fneu_session_mean, 'm-', linewidth=0.8, alpha=0.8, label='Session Fneu mean')
    ax.set_title('Session-wide Raw Signals')
    ax.set_ylabel('Fluorescence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(t_session, Fc_session_mean, 'b-', linewidth=0.8, alpha=0.8)
    ax.set_title('Session-wide Fc (Neuropil Corrected)')
    ax.set_ylabel('Fc')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.plot(t_session, F0_session_mean, 'orange', linewidth=0.8, alpha=0.8)
    ax.set_title('Session-wide F0 (Baseline)')
    ax.set_ylabel('F0')
    ax.grid(True, alpha=0.3)
    
    # Alpha distribution
    ax = axes[3]
    ax.hist(data['neuropil_a'], bins=50, alpha=0.7, color='red')
    ax.axvline(data['neuropil_a'].mean(), color='black', linestyle='--', 
              label=f'Mean: {data["neuropil_a"].mean():.3f}')
    ax.set_title('Alpha Coefficient Distribution')
    ax.set_xlabel('Neuropil Alpha')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Correlation analysis between F and Fneu for problematic ROIs
    print(f"\n=== F vs FNEU CORRELATION ANALYSIS ===")
    for roi in worst_rois[:3]:  # Top 3 worst
        F_roi = data['F'][roi]
        Fneu_roi = data['Fneu'][roi]
        corr = np.corrcoef(F_roi, Fneu_roi)[0, 1]
        
        # Check if Fneu ever exceeds F (impossible physically)
        fneu_exceeds_f = np.sum(Fneu_roi > F_roi)
        fneu_exceeds_pct = 100 * fneu_exceeds_f / len(F_roi)
        
        print(f"  ROI {roi}: F-Fneu correlation = {corr:.3f}")
        print(f"           Fneu > F: {fneu_exceeds_pct:.1f}% of timepoints")
        print(f"           Alpha: {data['neuropil_a'][roi]:.3f}")











def diagnose_problematic_roi_signals_comprehensive(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """Comprehensive diagnosis of problematic ROI signals with motion data and spatial analysis"""
    print("\n=== COMPREHENSIVE ROI SIGNAL DIAGNOSIS ===")
    
    # Find ALL problematic ROIs (not just top 5)
    dff_clean = data['dFF_clean']
    n_rois = dff_clean.shape[0]
    
    print(f"Analyzing ALL {n_rois} ROIs for issues...")
    
    # Comprehensive issue detection
    all_problem_rois = []
    issue_categories = {
        'exploded': [],       # |dF/F| > 50
        'extreme': [],        # |dF/F| > 10 or std > 3
        'negative_fc': [],    # >20% negative Fc values  
        'near_zero_f0': [],   # F0 < 0.1 frequently
        'high_alpha': [],     # Alpha > 0.65
        'signal_inversion': [] # Mean Fc < 0
    }
    
    for roi_idx in range(n_rois):
        roi_issues = []
        
        # dF/F explosion check
        roi_trace = dff_clean[roi_idx, :]
        roi_min = np.min(roi_trace)
        roi_max = np.max(roi_trace)
        roi_std = np.std(roi_trace)
        roi_range = roi_max - roi_min
        
        if roi_max > 50 or roi_min < -50:
            roi_issues.append('exploded')
            issue_categories['exploded'].append(roi_idx)
        elif roi_max > 10 or roi_min < -5 or roi_std > 3:
            roi_issues.append('extreme')
            issue_categories['extreme'].append(roi_idx)
        
        # Fc issues
        fc_roi = data['Fc'][roi_idx]
        negative_fc_pct = np.sum(fc_roi < 0) / len(fc_roi)
        if negative_fc_pct > 0.2:  # >20% negative
            roi_issues.append('negative_fc')
            issue_categories['negative_fc'].append(roi_idx)
        
        # F0 issues
        f0_roi = data['F0'][roi_idx]
        near_zero_f0_pct = np.sum(f0_roi < 0.1) / len(f0_roi)
        if near_zero_f0_pct > 0.1:  # >10% near zero
            roi_issues.append('near_zero_f0')
            issue_categories['near_zero_f0'].append(roi_idx)
        
        # Neuropil issues
        alpha_roi = data['neuropil_a'][roi_idx]
        if alpha_roi > 0.65:
            roi_issues.append('high_alpha')
            issue_categories['high_alpha'].append(roi_idx)
        
        # Signal inversion
        if np.mean(fc_roi) < 0:
            roi_issues.append('signal_inversion')
            issue_categories['signal_inversion'].append(roi_idx)
        
        if roi_issues:
            all_problem_rois.append({
                'roi': roi_idx,
                'issues': roi_issues,
                'severity_score': len(roi_issues) + (roi_range if roi_range < 1000 else 1000),
                'dff_range': roi_range,
                'negative_fc_pct': negative_fc_pct,
                'near_zero_f0_pct': near_zero_f0_pct,
                'alpha': alpha_roi
            })
    
    # Print comprehensive issue summary
    print(f"\n=== ISSUE DETECTION SUMMARY ===")
    print(f"Total ROIs analyzed: {n_rois}")
    print(f"ROIs with issues: {len(all_problem_rois)} ({100*len(all_problem_rois)/n_rois:.1f}%)")
    
    for category, roi_list in issue_categories.items():
        if len(roi_list) > 0:
            print(f"  {category}: {len(roi_list)} ROIs ({100*len(roi_list)/n_rois:.1f}%)")
    
    # Sort by severity and select for detailed analysis
    all_problem_rois.sort(key=lambda x: x['severity_score'], reverse=True)
    
    # Check motion data availability
    has_motion = False
    motion_x, motion_y = None, None
    if 'ops' in data and data['ops'] is not None:
        ops = data['ops']
        if 'xoff' in ops and 'yoff' in ops:
            motion_x = np.array(ops['xoff'])
            motion_y = np.array(ops['yoff'])
            has_motion = True
            print(f"\n=== MOTION DATA AVAILABLE ===")
            print(f"  Motion correction: {len(motion_x)} frames")
            print(f"  X drift range: [{motion_x.min():.2f}, {motion_x.max():.2f}] pixels")
            print(f"  Y drift range: [{motion_y.min():.2f}, {motion_y.max():.2f}] pixels")
            print(f"  X drift std: {motion_x.std():.2f} pixels")
            print(f"  Y drift std: {motion_y.std():.2f} pixels")
    
    # Generate diagnostic plots for problematic ROIs (6 per page in 3×2 layout)
    rois_per_page = 6
    n_pages = max(1, int(np.ceil(len(all_problem_rois) / rois_per_page)))
    
    print(f"\n=== GENERATING DIAGNOSTIC PLOTS ===")
    print(f"Creating {n_pages} diagnostic pages for {len(all_problem_rois)} problematic ROIs")
    
    save_dir = cfg['overlay']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    for page_idx in range(n_pages):
        start_idx = page_idx * rois_per_page
        end_idx = min((page_idx + 1) * rois_per_page, len(all_problem_rois))
        page_rois = all_problem_rois[start_idx:end_idx]
        
        print(f"  Page {page_idx + 1}/{n_pages}: ROIs {[r['roi'] for r in page_rois]}")
        
        # Create figure with 3×2 layout for 6 ROIs
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        axes = axes.flatten()  # Make indexing easier
        
        for plot_idx, roi_info in enumerate(page_rois):
            roi = roi_info['roi']
            issues = roi_info['issues']
            
            ax = axes[plot_idx]
            
            # Get all signals for this ROI
            F_roi = data['F'][roi]
            Fneu_roi = data['Fneu'][roi]
            Fc_roi = data['Fc'][roi]
            F0_roi = data['F0'][roi]
            dFF_roi = data['dFF_clean'][roi]
            alpha_roi = data['neuropil_a'][roi]
            
            # Time vector
            fs = cfg['acq']['fs']
            t = np.arange(len(F_roi)) / fs
            
            # Create subplot with multiple y-axes for comprehensive view
            ax2 = ax.twinx()
            ax3 = ax.twinx()
            ax3.spines['right'].set_position(('outward', 60))
            
            # Plot main signals
            l1 = ax.plot(t, F_roi, 'k-', linewidth=0.6, alpha=0.7, label='F')
            l2 = ax.plot(t, Fneu_roi, 'm-', linewidth=0.6, alpha=0.7, label='Fneu')
            l3 = ax2.plot(t, Fc_roi, 'b-', linewidth=0.6, alpha=0.8, label='Fc')
            l4 = ax2.plot(t, F0_roi, 'orange', linewidth=0.6, alpha=0.8, label='F0')
            l5 = ax3.plot(t, dFF_roi, 'purple', linewidth=0.6, alpha=0.9, label='dF/F')
            
            # Add horizontal reference lines
            ax2.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
            ax3.axhline(0, color='gray', linestyle='-', alpha=0.8, linewidth=0.8)
            
            # Color-code by issue severity
            if 'exploded' in issues:
                color = 'red'
            elif 'extreme' in issues:
                color = 'orange'
            elif len(issues) > 2:
                color = 'purple'
            else:
                color = 'blue'
            
            # Title with comprehensive info
            issue_str = ', '.join(issues)
            title = f"ROI {roi} ({color.upper()})\n{issue_str}\nα={alpha_roi:.3f}, range={roi_info['dff_range']:.1f}"
            ax.set_title(title, fontsize=10, color=color, weight='bold')
            
            # Set axis properties
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('F, Fneu', color='black')
            ax2.set_ylabel('Fc, F0', color='blue')
            ax3.set_ylabel('dF/F', color='purple')
            
            # Set reasonable y-limits to avoid extreme outliers compressing the plot
            ax3_lim = np.percentile(np.abs(dFF_roi[np.isfinite(dFF_roi)]), [99.5])[0]
            ax3.set_ylim(-ax3_lim*1.1, ax3_lim*1.1)
            
            # Add motion overlay if available
            if has_motion and len(motion_x) == len(t):
                # Normalize motion to small overlay
                motion_norm = (motion_x - motion_x.mean()) / (motion_x.std() + 1e-9) * 0.1 * ax3_lim
                ax3.fill_between(t, motion_norm, alpha=0.2, color='gray', label='Motion (norm)')
            
            ax.grid(True, alpha=0.3)
            
            # Combined legend
            lines = l1 + l2 + l3 + l4 + l5
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left', fontsize=8)
        
        # Hide unused subplots if we have fewer than 6 ROIs on this page
        for i in range(len(page_rois), 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save individual page
        save_path = os.path.join(save_dir, f'roi_signal_diagnosis_page_{page_idx+1:02d}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved: {save_path}")
        
        plt.close(fig)
    
    # Generate spatial analysis of problematic ROIs
    print(f"\n=== SPATIAL ANALYSIS OF ISSUES ===")
    
    if 'roi_features' in data and 'stat' in data:
        # Get ROI centroids
        feats = data['roi_features']
        
        # Create spatial issue map
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Show anatomy background
        if 'anatomy_image' in data:
            A = data['anatomy_image']
            ov_cfg = cfg['overlay']
            lo, hi = np.percentile(A, (ov_cfg['bg_pmin'], ov_cfg['bg_pmax']))
            bg = np.clip((A - lo) / (hi - lo + 1e-9), 0, 1)
            ax.imshow(bg, cmap='gray', alpha=0.5)
        
        # Plot all ROIs first (small gray dots)
        all_x = feats['centroid_x']
        all_y = feats['centroid_y']
        ax.scatter(all_x, all_y, c='lightgray', s=3, alpha=0.5, label='Normal ROIs')
        
        # Overlay problematic ROIs with color coding
        colors = {'exploded': 'red', 'extreme': 'orange', 'negative_fc': 'blue', 
                 'near_zero_f0': 'green', 'high_alpha': 'purple', 'signal_inversion': 'brown'}
        
        for category, roi_list in issue_categories.items():
            if len(roi_list) > 0:
                prob_x = [all_x[roi] for roi in roi_list]
                prob_y = [all_y[roi] for roi in roi_list]
                ax.scatter(prob_x, prob_y, c=colors[category], s=30, alpha=0.8, 
                          label=f'{category} (n={len(roi_list)})', edgecolors='black', linewidth=0.5)
        
        ax.set_title(f'Spatial Distribution of ROI Issues\n{len(all_problem_rois)} total problematic ROIs')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.invert_yaxis()  # Match image coordinates
        
        plt.tight_layout()
        
        # Save spatial analysis
        spatial_save_path = os.path.join(save_dir, 'roi_issues_spatial_map.png')
        fig.savefig(spatial_save_path, dpi=150, bbox_inches='tight')
        print(f"Spatial analysis saved: {spatial_save_path}")
        
        plt.close(fig)
    
    # Generate motion correlation analysis if available
    if has_motion:
        print(f"\n=== MOTION CORRELATION ANALYSIS ===")
        
        # Compute correlation between motion and signal issues
        motion_magnitude = np.sqrt(motion_x**2 + motion_y**2)
        
        # For each problematic ROI, compute correlation with motion
        motion_correlations = []
        
        for roi_info in all_problem_rois[:20]:  # Analyze top 20 most problematic
            roi = roi_info['roi']
            dff_roi = data['dFF_clean'][roi]
            
            if len(dff_roi) == len(motion_magnitude):
                # Compute correlation between |dF/F| and motion magnitude
                dff_abs = np.abs(dff_roi)
                corr = np.corrcoef(dff_abs, motion_magnitude)[0, 1]
                motion_correlations.append({
                    'roi': roi,
                    'motion_corr': corr,
                    'issues': roi_info['issues']
                })
        
        # Plot motion correlation results
        if motion_correlations:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Plot 1: Motion over time
            t_motion = np.arange(len(motion_x)) / fs
            ax1.plot(t_motion, motion_x, 'r-', linewidth=0.8, alpha=0.7, label='X motion')
            ax1.plot(t_motion, motion_y, 'b-', linewidth=0.8, alpha=0.7, label='Y motion')
            ax1.plot(t_motion, motion_magnitude, 'k-', linewidth=0.8, alpha=0.8, label='Magnitude')
            ax1.set_ylabel('Motion (pixels)')
            ax1.set_title('Motion Correction Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Motion correlation histogram
            corrs = [mc['motion_corr'] for mc in motion_correlations if not np.isnan(mc['motion_corr'])]
            ax2.hist(corrs, bins=20, alpha=0.7, color='purple')
            ax2.axvline(np.mean(corrs), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(corrs):.3f}')
            ax2.set_xlabel('Correlation with Motion')
            ax2.set_ylabel('Number of Problematic ROIs')
            ax2.set_title('Motion Correlation for Problematic ROIs')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            motion_save_path = os.path.join(save_dir, 'motion_correlation_analysis.png')
            fig.savefig(motion_save_path, dpi=150, bbox_inches='tight')
            print(f"Motion analysis saved: {motion_save_path}")
            
            plt.close(fig)
            
            # Print motion correlation summary
            high_motion_corr = [mc for mc in motion_correlations if abs(mc['motion_corr']) > 0.3]
            print(f"  ROIs with high motion correlation (|r| > 0.3): {len(high_motion_corr)}")
            
            if high_motion_corr:
                print(f"  Sample high-correlation ROIs:")
                for mc in high_motion_corr[:5]:
                    print(f"    ROI {mc['roi']}: r={mc['motion_corr']:.3f}, issues: {mc['issues']}")
    
    # Final summary
    print(f"\n=== COMPREHENSIVE DIAGNOSIS COMPLETE ===")
    print(f"Generated {n_pages} diagnostic pages")
    print(f"Spatial analysis: {len(all_problem_rois)} problematic ROIs mapped")
    if has_motion:
        print(f"Motion analysis: Available and analyzed")
    print(f"All plots saved to: {save_dir}")
    
    # Store comprehensive results in data for later use
    data['comprehensive_diagnosis'] = {
        'all_problem_rois': all_problem_rois,
        'issue_categories': issue_categories,
        'has_motion_data': has_motion,
        'motion_correlations': motion_correlations if has_motion else None,
        'total_rois': n_rois,
        'problem_roi_count': len(all_problem_rois)
    }
    
    return data


















































































def check_motion_data_availability(ops):
    """Check what motion data is available"""
    motion_keys = ['corrXY', 'badframes', 'xoff', 'yoff', 'rmin', 'rmax', 'nframes', 'frames_per_file']
    available = {key: key in ops and ops[key] is not None for key in motion_keys}
    
    print(f"Motion data availability:")
    for key, avail in available.items():
        print(f"  {key}: {'✓' if avail else '✗'}")
    
    return available

def assess_registration_quality(ops):
    """Assess overall registration quality"""
    
    corrXY = ops.get('corrXY')
    xoff = ops.get('xoff')
    yoff = ops.get('yoff')
    
    if corrXY is None or xoff is None or yoff is None:
        print("Insufficient motion data for quality assessment")
        return {}, []
    
    motion_mag = np.sqrt(np.array(xoff)**2 + np.array(yoff)**2)
    
    quality_metrics = {
        'mean_correlation': np.mean(corrXY),
        'correlation_stability': np.std(corrXY),
        'poor_frames_pct': 100 * np.sum(corrXY < 0.8) / len(corrXY),
        'max_motion_px': np.max(motion_mag),
        'motion_rms': np.sqrt(np.mean(motion_mag**2)),
        'large_motion_frames': np.sum(motion_mag > 2.0),  # >2 pixel motion
    }
    
    # Quality flags
    flags = []
    if quality_metrics['mean_correlation'] < 0.85:
        flags.append('low_correlation')
    if quality_metrics['poor_frames_pct'] > 5:
        flags.append('many_poor_frames')
    if quality_metrics['max_motion_px'] > 10:
        flags.append('excessive_motion')
    if quality_metrics['motion_rms'] > 1.0:
        flags.append('high_motion_variance')
    
    return quality_metrics, flags



def check_file_boundary_artifacts(data, ops):
    """Check for signal jumps near file boundaries - FIXED for array frames_per_file"""
    frames_per_file = ops.get('frames_per_file')
    nframes = ops.get('nframes')
    
    if frames_per_file is None or nframes is None:
        return []
    
    # FIXED: Handle array of frame counts
    if isinstance(frames_per_file, (list, np.ndarray)) and len(frames_per_file) > 1:
        # Calculate cumulative file boundaries
        file_boundaries = np.cumsum(frames_per_file)[:-1]  # Exclude last boundary
    elif isinstance(frames_per_file, (int, np.integer)) or (isinstance(frames_per_file, np.ndarray) and frames_per_file.size == 1):
        # Single value case
        frames_per_file_val = int(frames_per_file)
        file_boundaries = np.arange(frames_per_file_val, nframes, frames_per_file_val)
    else:
        return []
    
    # Check for signal jumps near boundaries
    boundary_affected_rois = []
    dff_clean = data.get('dFF_clean')
    
    if dff_clean is None:
        return []
    
    for roi_idx in range(dff_clean.shape[0]):
        for boundary in file_boundaries:
            # Check ±10 frames around boundary
            if boundary > 10 and boundary < dff_clean.shape[1] - 10:
                pre_signal = dff_clean[roi_idx, boundary-10:boundary]
                post_signal = dff_clean[roi_idx, boundary:boundary+10]
                
                signal_jump = abs(np.mean(post_signal) - np.mean(pre_signal))
                if signal_jump > 0.5:  # Significant jump
                    boundary_affected_rois.append({
                        'roi': roi_idx,
                        'boundary_frame': boundary,
                        'signal_jump': signal_jump
                    })
    
    return boundary_affected_rois


def check_roi_near_registration_boundaries(data, ops, margin_px=10):
    """Check if ROI touches registration boundaries"""
    yrange = ops.get('yrange')
    xrange = ops.get('xrange')
    
    if yrange is None or xrange is None:
        return []
    
    boundary_rois = []
    for roi_idx, stat_entry in enumerate(data['stat']):
        ypix = stat_entry['ypix']
        xpix = stat_entry['xpix']
        
        near_y_boundary = (ypix.min() < yrange[0] + margin_px) or (ypix.max() > yrange[1] - margin_px)
        near_x_boundary = (xpix.min() < xrange[0] + margin_px) or (xpix.max() > xrange[1] - margin_px)
        
        if near_y_boundary or near_x_boundary:
            boundary_rois.append(roi_idx)
    
    return boundary_rois

def correlate_motion_with_signals(data, ops):
    """Correlate signal with motion and frame quality"""
    corrXY = ops.get('corrXY')
    xoff = ops.get('xoff')
    yoff = ops.get('yoff')
    
    if corrXY is None or xoff is None or yoff is None:
        return []
    
    motion_mag = np.sqrt(np.array(xoff)**2 + np.array(yoff)**2)
    dff_clean = data.get('dFF_clean')
    
    if dff_clean is None:
        return []
    
    correlations = []
    for roi_idx in range(dff_clean.shape[0]):
        dff_abs = np.abs(dff_clean[roi_idx])
        
        if len(dff_abs) == len(motion_mag):
            # Correlate signal with motion and frame quality
            motion_corr = np.corrcoef(dff_abs, motion_mag)[0,1] if not np.any(np.isnan(dff_abs)) else np.nan
            quality_corr = np.corrcoef(dff_abs, corrXY)[0,1] if not np.any(np.isnan(dff_abs)) else np.nan
            
            correlations.append({
                'roi': roi_idx,
                'motion_correlation': motion_corr,
                'quality_correlation': quality_corr
            })
    
    return correlations

def calculate_severity_score(data, roi_idx):
    """Calculate severity score for ROI issues"""
    score = 0
    issues = []
    
    # dF/F explosion (highest severity)
    if 'dFF_clean' in data:
        dff = data['dFF_clean'][roi_idx]
        dff_max = np.max(dff)
        dff_min = np.min(dff)
        dff_range = dff_max - dff_min
        
        if dff_max > 100 or dff_min < -100:
            score += 1000  # Extreme explosion
            issues.append('extreme_explosion')
        elif dff_max > 50 or dff_min < -50:
            score += 500   # Major explosion
            issues.append('major_explosion')
        elif dff_max > 10 or dff_min < -5:
            score += 100   # Moderate explosion
            issues.append('moderate_explosion')
    
    # Negative Fc (signal inversion)
    if 'Fc' in data:
        fc = data['Fc'][roi_idx]
        negative_pct = np.sum(fc < 0) / len(fc)
        if negative_pct > 0.5:
            score += 200
            issues.append('major_signal_inversion')
        elif negative_pct > 0.2:
            score += 50
            issues.append('moderate_signal_inversion')
    
    # Near-zero F0 (baseline failure)
    if 'F0' in data:
        f0 = data['F0'][roi_idx]
        near_zero_pct = np.sum(f0 < 0.1) / len(f0)
        if near_zero_pct > 0.3:
            score += 150
            issues.append('baseline_failure')
        elif near_zero_pct > 0.1:
            score += 30
            issues.append('baseline_issues')
    
    # High alpha (excessive neuropil)
    if 'neuropil_a' in data:
        alpha = data['neuropil_a'][roi_idx]
        if alpha > 0.7:
            score += 75
            issues.append('excessive_neuropil')
        elif alpha > 0.65:
            score += 25
            issues.append('high_neuropil')
    
    return score, issues

def identify_problematic_rois(data, max_rois=50):
    """Identify most problematic ROIs by severity"""
    n_rois = data['F'].shape[0]
    roi_scores = []
    
    for roi_idx in range(n_rois):
        score, issues = calculate_severity_score(data, roi_idx)
        if score > 0:  # Only include ROIs with issues
            roi_scores.append({
                'roi': roi_idx,
                'severity_score': score,
                'issues': issues
            })
    
    # Sort by severity and return top cases
    roi_scores.sort(key=lambda x: x['severity_score'], reverse=True)
    return roi_scores[:max_rois]

def plot_comprehensive_roi_diagnostic(data: Dict[str, Any], cfg: Dict[str, Any], 
                                    roi_idx: int, save: bool = False) -> Any:
    """Plot comprehensive diagnostic for a single ROI with all processing stages and motion data"""
    print(f"\n=== COMPREHENSIVE DIAGNOSTIC - ROI {roi_idx} ===")
    
    # Check data availability
    required_keys = ['F', 'Fneu', 'Fc', 'F0', 'dFF_clean']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required data: {key}")
    
    # Get ops data for motion
    ops = data.get('ops', {})
    has_motion = all(key in ops for key in ['corrXY', 'xoff', 'yoff'])
    has_file_info = 'frames_per_file' in ops and 'nframes' in ops
    
    fs = cfg['acq']['fs']
    n_frames = len(data['F'][roi_idx])
    t = np.arange(n_frames) / fs
    
    # Create comprehensive figure (8 subplots)
    fig, axes = plt.subplots(8, 1, figsize=(16, 24), sharex=True)
    
    # Get all signals for this ROI
    F_roi = data['F'][roi_idx]
    Fneu_roi = data['Fneu'][roi_idx]
    Fc_roi = data['Fc'][roi_idx]
    F0_roi = data['F0'][roi_idx]
    dFF_roi = data['dFF'][roi_idx] if 'dFF' in data else data['dFF_clean'][roi_idx]
    dFF_clean_roi = data['dFF_clean'][roi_idx]
    alpha_roi = data['neuropil_a'][roi_idx] if 'neuropil_a' in data else 0.7
    
    # Calculate severity and issues
    severity_score, issues = calculate_severity_score(data, roi_idx)
    
    # 1. F and Fneu comparison
    ax = axes[0]
    ax.plot(t, F_roi, 'k-', linewidth=0.6, alpha=0.8, label='F')
    ax.plot(t, Fneu_roi, 'm-', linewidth=0.6, alpha=0.8, label='Fneu')
    
    # Check for Fneu > F (problematic)
    fneu_exceeds = np.sum(Fneu_roi > F_roi)
    fneu_exceeds_pct = 100 * fneu_exceeds / len(F_roi)
    
    ax.set_title(f'ROI {roi_idx} - Raw F & Fneu\nα={alpha_roi:.3f}, Fneu>F: {fneu_exceeds_pct:.1f}%')
    ax.set_ylabel('Fluorescence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Neuropil correction breakdown
    ax = axes[1]
    ax.plot(t, F_roi, 'k-', linewidth=0.6, alpha=0.8, label='F')
    ax.plot(t, alpha_roi * Fneu_roi, 'r-', linewidth=0.6, alpha=0.8, label=f'{alpha_roi:.2f}×Fneu')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Neuropil Correction: F vs α×Fneu')
    ax.set_ylabel('Fluorescence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Fc quality check
    ax = axes[2]
    ax.plot(t, Fc_roi, 'b-', linewidth=0.6, alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    negative_fc_pct = 100 * np.sum(Fc_roi < 0) / len(Fc_roi)
    fc_range = np.max(Fc_roi) - np.min(Fc_roi)
    
    ax.set_title(f'Fc (Neuropil Corrected)\nRange: {fc_range:.1f}, Negative: {negative_fc_pct:.1f}%')
    ax.set_ylabel('Fc')
    ax.grid(True, alpha=0.3)
    
    # 4. F0 baseline with failure detection
    ax = axes[3]
    ax.plot(t, Fc_roi, 'b-', linewidth=0.4, alpha=0.5, label='Fc')
    ax.plot(t, F0_roi, 'orange', linewidth=0.8, alpha=0.8, label='F0')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    near_zero_f0_pct = 100 * np.sum(F0_roi < 0.1) / len(F0_roi)
    negative_f0_pct = 100 * np.sum(F0_roi < 0) / len(F0_roi)
    
    ax.set_title(f'F0 Baseline\nNear-zero: {near_zero_f0_pct:.1f}%, Negative: {negative_f0_pct:.1f}%')
    ax.set_ylabel('Fc & F0')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 5. dF/F calculation breakdown
    ax = axes[4]
    numerator = Fc_roi - F0_roi
    ax.plot(t, numerator, 'g-', linewidth=0.6, alpha=0.8, label='Fc-F0 (numerator)')
    ax.plot(t, F0_roi, 'orange', linewidth=0.6, alpha=0.8, label='F0 (denominator)')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('dF/F Components: (Fc-F0) and F0')
    ax.set_ylabel('ΔF, F0')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 6. Final dF/F result
    ax = axes[5]
    ax.plot(t, dFF_clean_roi, 'purple', linewidth=0.6, alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    dff_min = np.min(dFF_clean_roi)
    dff_max = np.max(dFF_clean_roi)
    dff_range = dff_max - dff_min
    
    # Mark explosions
    explosion_mask = (dFF_clean_roi > 10) | (dFF_clean_roi < -5)
    if np.any(explosion_mask):
        ax.scatter(t[explosion_mask], dFF_clean_roi[explosion_mask], 
                  c='red', s=10, alpha=0.8, zorder=5)
    
    ax.set_title(f'Final dF/F\nRange: [{dff_min:.1f}, {dff_max:.1f}], Explosions: {np.sum(explosion_mask)}')
    ax.set_ylabel('dF/F')
    ax.grid(True, alpha=0.3)
    
    # 7. Motion quality (if available)
    if has_motion:
        ax = axes[6]
        
        corrXY = np.array(ops['corrXY'])
        xoff = np.array(ops['xoff'])
        yoff = np.array(ops['yoff'])
        motion_mag = np.sqrt(xoff**2 + yoff**2)
        
        # Ensure motion data matches imaging data length
        if len(corrXY) == len(t):
            ax.plot(t, corrXY, 'orange', linewidth=0.6, alpha=0.8, label='Frame correlation')
            ax.axhline(corrXY.mean() - 2*corrXY.std(), color='red', linestyle='--', 
                      alpha=0.7, label='Poor quality threshold')
            
            # Mark bad frames if available
            badframes = ops.get('badframes')
            if badframes is not None and len(badframes) == len(t):
                bad_mask = np.array(badframes, dtype=bool)
                if np.any(bad_mask):
                    ax.scatter(t[bad_mask], corrXY[bad_mask], 
                             c='red', s=15, marker='x', alpha=0.8, zorder=5)
            
            # Motion magnitude on second y-axis
            ax2 = ax.twinx()
            ax2.plot(t, motion_mag, 'gray', linewidth=0.4, alpha=0.6, label='Motion (px)')
            ax2.set_ylabel('Motion (px)', color='gray')
            
            ax.set_title('Motion Quality: Frame Correlation & Motion Magnitude')
            ax.set_ylabel('Correlation')
            ax.legend(loc='upper left', fontsize=8)
            ax2.legend(loc='upper right', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Motion data length mismatch', transform=ax.transAxes, 
                   ha='center', va='center')
            ax.set_title('Motion Quality: Data Unavailable')
    else:
        ax = axes[6]
        ax.text(0.5, 0.5, 'Motion data not available', transform=ax.transAxes, 
               ha='center', va='center')
        ax.set_title('Motion Quality: Data Unavailable')
    
    ax.grid(True, alpha=0.3)
    
    # 8. File boundary context (if available)
    ax = axes[7]
    
    if has_file_info:
        frames_per_file = ops['frames_per_file']
        nframes = ops['nframes']
        
        # Plot a representative signal
        ax.plot(t, dFF_clean_roi, 'purple', linewidth=0.6, alpha=0.8, label='dF/F')
        
        # FIXED: Calculate cumulative file boundaries from frame counts array
        if isinstance(frames_per_file, (list, np.ndarray)) and len(frames_per_file) > 1:
            # frames_per_file is an array of frame counts per file
            # Calculate cumulative boundaries
            file_boundaries = np.cumsum(frames_per_file)[:-1]  # Exclude last boundary (end of recording)
            
            print(f"    File structure: {len(frames_per_file)} files with frame counts: {frames_per_file}")
            print(f"    File boundaries at frames: {file_boundaries}")
            
        elif isinstance(frames_per_file, (int, np.integer)) or (isinstance(frames_per_file, np.ndarray) and frames_per_file.size == 1):
            # Single value case (all files have same frame count)
            frames_per_file_val = int(frames_per_file)
            file_boundaries = np.arange(frames_per_file_val, nframes, frames_per_file_val)
            
        else:
            # Fallback - no boundaries to plot
            file_boundaries = []
            print(f"    Warning: Cannot interpret frames_per_file: {frames_per_file}")
        
        # Mark file boundaries
        for boundary_frame in file_boundaries:
            if boundary_frame < len(t):
                boundary_time = boundary_frame / fs
                ax.axvline(boundary_time, color='purple', linestyle=':', alpha=0.8, linewidth=2)
                ax.text(boundary_time, ax.get_ylim()[1]*0.9, 'File\nBoundary', 
                    rotation=90, ha='right', va='top', fontsize=7, color='purple')
        
        ax.set_title(f'File Boundaries ({len(frames_per_file)} files, boundaries at: {file_boundaries})')
    else:
        ax.plot(t, dFF_clean_roi, 'purple', linewidth=0.6, alpha=0.8)
        ax.set_title('File Boundary Info: Not Available')

    ax.set_ylabel('dF/F')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    
    # Add overall severity assessment
    severity_text = f"SEVERITY: {severity_score} | ISSUES: {', '.join(issues) if issues else 'None'}"
    if severity_score > 500:
        severity_color = 'red'
    elif severity_score > 100:
        severity_color = 'orange'
    elif severity_score > 0:
        severity_color = 'blue'
    else:
        severity_color = 'green'
    
    fig.suptitle(f'Comprehensive ROI Diagnostic - {severity_text}', 
                 fontsize=14, color=severity_color, weight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if save:
        save_dir = cfg['overlay']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'comprehensive_diagnostic_roi_{roi_idx:04d}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comprehensive diagnostic saved: {save_path}")
    
    return fig

def generate_comprehensive_diagnostics(data: Dict[str, Any], cfg: Dict[str, Any], 
                                     max_rois: int = 20, save: bool = True) -> None:
    """Generate comprehensive diagnostics for the most problematic ROIs"""
    print("\n=== GENERATING COMPREHENSIVE DIAGNOSTICS ===")
    
    # Check motion data availability
    ops = data.get('ops', {})
    motion_available = check_motion_data_availability(ops)
    
    if motion_available['corrXY'] and motion_available['xoff'] and motion_available['yoff']:
        print("\n=== MOTION DATA ANALYSIS ===")
        quality_metrics, quality_flags = assess_registration_quality(ops)
        
        print("Registration quality metrics:")
        for metric, value in quality_metrics.items():
            print(f"  {metric}: {value:.3f}")
        
        if quality_flags:
            print(f"Quality warnings: {', '.join(quality_flags)}")
        else:
            print("Registration quality: GOOD")
        
        # Motion-signal correlations
        motion_correlations = correlate_motion_with_signals(data, ops)
        if motion_correlations:
            high_motion_corr = [mc for mc in motion_correlations if abs(mc.get('motion_correlation', 0)) > 0.3]
            print(f"ROIs with high motion correlation: {len(high_motion_corr)}")
    
    # File boundary analysis
    if motion_available['frames_per_file'] and motion_available['nframes']:
        boundary_artifacts = check_file_boundary_artifacts(data, ops)
        if boundary_artifacts:
            print(f"File boundary artifacts detected: {len(boundary_artifacts)} cases")
        else:
            print("No file boundary artifacts detected")
    
    # Registration boundary analysis
    if motion_available.get('yrange') and motion_available.get('xrange'):
        boundary_rois = check_roi_near_registration_boundaries(data, ops)
        if boundary_rois:
            print(f"ROIs near registration boundaries: {len(boundary_rois)}")
        else:
            print("No ROIs near registration boundaries")
    
    # Identify most problematic ROIs
    print("\n=== IDENTIFYING PROBLEMATIC ROIS ===")
    problematic_rois = identify_problematic_rois(data, max_rois=max_rois)
    
    if not problematic_rois:
        print("No significantly problematic ROIs found!")
        return
    
    print(f"Found {len(problematic_rois)} problematic ROIs for detailed analysis")
    print("Top 10 most severe cases:")
    for i, roi_info in enumerate(problematic_rois[:10]):
        roi_idx = roi_info['roi']
        score = roi_info['severity_score']
        issues = ', '.join(roi_info['issues'])
        print(f"  {i+1:2d}. ROI {roi_idx:3d}: Score {score:4d} | {issues}")
    
    # Generate detailed diagnostics for most severe cases
    print(f"\n=== GENERATING DETAILED DIAGNOSTICS ===")
    print(f"Creating comprehensive diagnostic plots for top {min(max_rois, len(problematic_rois))} ROIs...")
    
    for i, roi_info in enumerate(problematic_rois):
        roi_idx = roi_info['roi']
        
        if i % 5 == 0:
            print(f"  Progress: {i+1}/{len(problematic_rois)} - ROI {roi_idx}")
        
        try:
            fig = plot_comprehensive_roi_diagnostic(data, cfg, roi_idx, save=save)
            if not save:  # Only show if not saving (to avoid too many windows)
                plt.show()
            plt.close(fig)
            
        except Exception as e:
            print(f"  Error plotting ROI {roi_idx}: {e}")
    
    # Generate summary report
    print(f"\n=== DIAGNOSTIC SUMMARY ===")
    
    issue_counts = {}
    for roi_info in problematic_rois:
        for issue in roi_info['issues']:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    print("Issue frequency:")
    for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {issue}: {count} ROIs")
    
    # Severity distribution
    severity_scores = [roi_info['severity_score'] for roi_info in problematic_rois]
    print(f"\nSeverity distribution:")
    print(f"  Extreme (>500): {sum(1 for s in severity_scores if s > 500)}")
    print(f"  Major (100-500): {sum(1 for s in severity_scores if 100 <= s <= 500)}")
    print(f"  Moderate (50-100): {sum(1 for s in severity_scores if 50 <= s < 100)}")
    print(f"  Minor (<50): {sum(1 for s in severity_scores if s < 50)}")
    
    print(f"\nRecommendations:")
    if issue_counts.get('extreme_explosion', 0) + issue_counts.get('major_explosion', 0) > 5:
        print("  - Consider baseline protection (minimum F0 threshold)")
        print("  - Check for photobleaching or movement artifacts")
    
    if issue_counts.get('major_signal_inversion', 0) > 10:
        print("  - Review neuropil correction parameters")
        print("  - Consider spatial filtering of Suite2p ROIs")
    
    if motion_available['corrXY'] and quality_flags:
        print("  - Motion correction quality issues detected")
        print("  - Consider re-running Suite2p with different registration parameters")
    
    print(f"\nComprehensive diagnostics complete!")
    if save:
        save_dir = cfg['overlay']['save_dir']
        print(f"All diagnostic plots saved to: {save_dir}")

# Add convenience function to the main pipeline
def run_comprehensive_diagnostics(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """Run comprehensive diagnostics for problematic ROIs"""
    generate_comprehensive_diagnostics(data, cfg, max_rois=20, save=True)



























def diagnose_roi_explosion_types(data: Dict[str, Any], roi_list: list = None) -> None:
    """Distinguish between neuropil vs. dynamic range explosion types"""
    print("\n=== DIAGNOSING EXPLOSION TYPES ===")
    
    if roi_list is None:
        # Find all problematic ROIs
        roi_list = []
        for roi in range(data['dFF_clean'].shape[0]):
            dff = data['dFF_clean'][roi]
            if np.max(dff) > 10 or np.min(dff) < -5:
                roi_list.append(roi)
        print(f"Found {len(roi_list)} problematic ROIs: {roi_list}")
    
    for roi in roi_list:
        print(f"\n--- ROI {roi} EXPLOSION TYPE ANALYSIS ---")
        
        F_roi = data['F'][roi]
        Fneu_roi = data['Fneu'][roi]
        Fc_roi = data['Fc'][roi]
        F0_roi = data['F0'][roi]
        alpha_roi = data['neuropil_a'][roi]
        
        # Type 1 indicators: Neuropil contamination
        fneu_exceeds_f = np.sum(Fneu_roi > F_roi) / len(F_roi)
        fc_negative = np.sum(Fc_roi < 0) / len(Fc_roi)
        f0_near_zero = np.sum(F0_roi < 0.1) / len(F0_roi)
        high_alpha = alpha_roi > 0.65
        
        # Type 2 indicators: Large dynamic range
        fc_range = np.max(Fc_roi) - np.min(Fc_roi)
        fc_positive_range = np.max(Fc_roi) - np.mean(Fc_roi[Fc_roi > 0])
        f0_follows_fc = np.corrcoef(Fc_roi, F0_roi)[0,1] if len(Fc_roi) > 1 else 0
        
        # Relative F0 variation during high Fc periods
        high_fc_mask = Fc_roi > np.percentile(Fc_roi, 90)
        if np.any(high_fc_mask):
            f0_during_high_fc = F0_roi[high_fc_mask]
            fc_during_high_fc = Fc_roi[high_fc_mask]
            f0_relative_var = np.std(f0_during_high_fc) / np.mean(f0_during_high_fc)
            fc_f0_ratio_high = np.mean(fc_during_high_fc / f0_during_high_fc)
        else:
            f0_relative_var = 0
            fc_f0_ratio_high = 1
        
        print(f"  Type 1 (Neuropil) indicators:")
        print(f"    Fneu > F: {100*fneu_exceeds_f:.1f}%")
        print(f"    Fc negative: {100*fc_negative:.1f}%") 
        print(f"    F0 near zero: {100*f0_near_zero:.1f}%")
        print(f"    High alpha (>{0.65}): {high_alpha} (α={alpha_roi:.3f})")
        
        print(f"  Type 2 (Dynamic Range) indicators:")
        print(f"    Fc range: {fc_range:.1f}")
        print(f"    Fc-F0 correlation: {f0_follows_fc:.3f}")
        print(f"    F0 relative variation during high Fc: {100*f0_relative_var:.1f}%")
        print(f"    Fc/F0 ratio during high activity: {fc_f0_ratio_high:.1f}x")
        
        # Classification
        type1_score = (fneu_exceeds_f > 0.1) + (fc_negative > 0.1) + (f0_near_zero > 0.1) + high_alpha
        type2_score = (fc_range > 500) + (f0_follows_fc > 0.8) + (fc_f0_ratio_high > 5) + (f0_relative_var < 0.3)
        
        if type1_score >= 2:
            explosion_type = "TYPE 1: NEUROPIL CONTAMINATION"
            recommendations = [
                "- Review ROI segmentation quality",
                "- Consider spatial filtering",
                "- Check if ROI includes blood vessels",
                "- Try different neuropil correction parameters"
            ]
        elif type2_score >= 2:
            explosion_type = "TYPE 2: LARGE DYNAMIC RANGE"
            recommendations = [
                "- Consider capping dF/F values (e.g., clip at ±10)",
                "- Use relative change metrics instead of dF/F",
                "- Apply smoothing to F0 baseline",
                "- Check if this represents genuine large Ca²⁺ transients"
            ]
        else:
            explosion_type = "MIXED OR UNCLEAR"
            recommendations = [
                "- Needs manual inspection",
                "- Check raw fluorescence traces",
                "- Consider both neuropil and baseline issues"
            ]
        
        print(f"  CLASSIFICATION: {explosion_type}")
        print(f"  Recommendations:")
        for rec in recommendations:
            print(f"    {rec}")












































def flag_roi_quality(fc, f0):
    flags = {}
    
    # Flag 1: High percentage of negative values
    neg_percentage = (fc < 0).sum() / len(fc) * 100
    flags['high_negative'] = neg_percentage > 20  # threshold: 20%
    
    # Flag 2: Poor SNR
    signal_std = np.std(fc)
    noise_estimate = np.std(fc[fc < np.percentile(fc, 25)])  # noise from low values
    snr = signal_std / noise_estimate if noise_estimate > 0 else 0
    flags['low_snr'] = snr < 2  # threshold: SNR < 2
    
    # Flag 3: Unstable baseline (high baseline variability)
    baseline_variability = np.std(np.percentile(fc.reshape(-1, 100), 10, axis=1))
    flags['unstable_baseline'] = baseline_variability > 0.1 * np.std(fc)
    
    # Flag 4: Extreme dynamic range
    dynamic_range = (np.percentile(fc, 95) - np.percentile(fc, 5)) / np.mean(f0)
    flags['extreme_range'] = dynamic_range > 2.0  # threshold: 200% range
    
    return flags


def calculate_robust_dff(f, method='percentile'):
    if method == 'percentile':
        # Use percentile-based F0 (more robust than mean)
        f0 = np.percentile(f, 10)  # or sliding percentile
        
    elif method == 'sliding_percentile':
        # Sliding window percentile baseline
        window_size = min(3000, len(f) // 10)  # adaptive window
        f0_series = pd.Series(f).rolling(window_size, center=True).quantile(0.1)
        f0_series = f0_series.interpolate().fillna(method='bfill').fillna(method='ffill')
        return (f - f0_series) / f0_series
        
    elif method == 'mode_robust':
        # Use mode of distribution as baseline
        hist, bin_edges = np.histogram(f, bins=50)
        f0 = bin_edges[np.argmax(hist)]
        
    # Clip extreme values to prevent explosion
    dff = (f - f0) / f0
    dff = np.clip(dff, -2, 5)  # reasonable physiological limits
    
    return dff






def adaptive_baseline_dff(f, tau_fast=1.0, tau_slow=30.0, fs=30):
    """
    Dual-timescale baseline tracking
    tau_fast, tau_slow in seconds
    """
    alpha_fast = 1 - np.exp(-1/(tau_fast * fs))
    alpha_slow = 1 - np.exp(-1/(tau_slow * fs))
    
    f0_fast = np.zeros_like(f)
    f0_slow = np.zeros_like(f)
    
    f0_fast[0] = f[0]
    f0_slow[0] = f[0]
    
    for i in range(1, len(f)):
        # Fast baseline tracks decreases quickly
        if f[i] < f0_fast[i-1]:
            f0_fast[i] = alpha_fast * f[i] + (1-alpha_fast) * f0_fast[i-1]
        else:
            f0_fast[i] = 0.1 * alpha_fast * f[i] + (1-0.1*alpha_fast) * f0_fast[i-1]
            
        # Slow baseline for overall trend
        f0_slow[i] = alpha_slow * f[i] + (1-alpha_slow) * f0_slow[i-1]
    
    # Use minimum of both baselines
    f0 = np.minimum(f0_fast, f0_slow)
    
    # Add small offset to prevent division by very small numbers
    f0_safe = np.maximum(f0, 0.1 * np.percentile(f, 50))
    
    return (f - f0_safe) / f0_safe




# Method 1: Percentile-based baseline
def correct_baseline_percentile(fc, percentile=10):
    baseline_offset = np.percentile(fc, percentile)
    fc_corrected = fc - baseline_offset
    return fc_corrected

# Method 2: Rolling minimum baseline
def correct_baseline_rolling_min(fc, window_size=1000):
    baseline = pd.Series(fc).rolling(window_size, center=True).min()
    baseline = baseline.interpolate().fillna(method='bfill').fillna(method='ffill')
    return fc - baseline

# Method 3: Robust baseline using median filter
from scipy.signal import medfilt
def correct_baseline_robust(fc, kernel_size=501):
    baseline = medfilt(fc, kernel_size=kernel_size)
    return fc - baseline





def process_roi_data(f_raw):
    # Step 1: Flag ROI quality
    flags = flag_roi_quality(f_raw, np.mean(f_raw))
    
    # Step 2: Choose processing based on flags
    if flags['extreme_range'] or flags['unstable_baseline']:
        # Use adaptive baseline for problematic ROIs
        dff = adaptive_baseline_dff(f_raw)
        processing_method = 'adaptive'
        
    elif flags['high_negative']:
        # Apply baseline correction first
        f_corrected = correct_baseline_percentile(f_raw, percentile=5)
        dff = calculate_robust_dff(f_corrected, method='percentile')
        processing_method = 'baseline_corrected'
        
    else:
        # Standard processing for good ROIs
        dff = calculate_robust_dff(f_raw, method='percentile')
        processing_method = 'standard'
    
    # Step 3: Final quality check
    final_flags = flag_roi_quality(dff * 100, 100)  # convert back to percentage scale
    
    return {
        'dff': dff,
        'flags': flags,
        'final_flags': final_flags,
        'processing_method': processing_method,
        'quality_score': calculate_quality_score(flags)
    }

def calculate_quality_score(flags):
    """Simple quality score: 0 (bad) to 1 (excellent)"""
    penalty = sum([
        flags['high_negative'] * 0.3,
        flags['low_snr'] * 0.4,
        flags['unstable_baseline'] * 0.2,
        flags['extreme_range'] * 0.1
    ])
    return max(0, 1 - penalty)










































# In img_main.py - simple ROI quality tracking

def initialize_roi_quality_df(n_rois: int) -> pd.DataFrame:
    """Initialize DataFrame to track ROI measurements and flags"""
    return pd.DataFrame({
        'roi_id': range(n_rois),
        # Will add measurements and flags as we go
    }).set_index('roi_id')

def add_roi_measurements(roi_df: pd.DataFrame, **measurements) -> pd.DataFrame:
    """Add measurement columns to ROI DataFrame"""
    for name, values in measurements.items():
        if len(values) != len(roi_df):
            raise ValueError(f"Length mismatch for {name}: {len(values)} vs {len(roi_df)}")
        roi_df[name] = values
    return roi_df

def add_roi_flags(roi_df: pd.DataFrame, **flags) -> pd.DataFrame:
    """Add flag columns to ROI DataFrame"""
    for name, values in flags.items():
        if np.isscalar(values):
            roi_df[name] = values  # Broadcast scalar
        else:
            if len(values) != len(roi_df):
                raise ValueError(f"Length mismatch for {name}: {len(values)} vs {len(roi_df)}")
            roi_df[name] = values
    return roi_df

# Usage in your pipeline:
def process_rois_with_quality_tracking(data: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Process ROIs and track quality metrics"""
    
    F = data['F']
    Fneu = data['Fneu']
    n_rois = F.shape[0]
    
    # Initialize quality tracking
    roi_quality = initialize_roi_quality_df(n_rois)
    
    # Add basic measurements
    roi_quality = add_roi_measurements(
        roi_quality,
        f_mean=np.mean(F, axis=1),
        f_std=np.std(F, axis=1),
        f_max=np.max(F, axis=1),
        f_min=np.min(F, axis=1),
        fneu_mean=np.mean(Fneu, axis=1),
        fneu_max=np.max(Fneu, axis=1),
        alpha=data['neuropil_a'] if 'neuropil_a' in data else np.full(n_rois, 0.7)
    )
    
    # Add derived measurements
    roi_quality = add_roi_measurements(
        roi_quality,
        f_range=roi_quality['f_max'] - roi_quality['f_min'],
        f_snr=roi_quality['f_mean'] / roi_quality['f_std'],
        fneu_f_ratio=roi_quality['fneu_mean'] / roi_quality['f_mean']
    )
    
    # Add flags based on measurements
    roi_quality = add_roi_flags(
        roi_quality,
        high_alpha=roi_quality['alpha'] > 0.65,
        low_snr=roi_quality['f_snr'] < 2.0,
        fneu_exceeds_f=roi_quality['fneu_f_ratio'] > 0.8,
        extreme_range=roi_quality['f_range'] > roi_quality['f_mean'] * 3
    )
    
    # Process Fc and add more measurements/flags
    if 'Fc' in data:
        Fc = data['Fc']
        roi_quality = add_roi_measurements(
            roi_quality,
            fc_mean=np.mean(Fc, axis=1),
            fc_min=np.min(Fc, axis=1),
            fc_negative_pct=np.mean(Fc < 0, axis=1) * 100
        )
        
        roi_quality = add_roi_flags(
            roi_quality,
            fc_negative=roi_quality['fc_min'] < 0,
            high_fc_negative=roi_quality['fc_negative_pct'] > 20
        )
    
    # Process dF/F and add final flags
    if 'dFF_clean' in data:
        dFF = data['dFF_clean']
        roi_quality = add_roi_measurements(
            roi_quality,
            dff_max=np.max(dFF, axis=1),
            dff_min=np.min(dFF, axis=1),
            dff_std=np.std(dFF, axis=1)
        )
        
        roi_quality = add_roi_flags(
            roi_quality,
            dff_explosion=np.abs(roi_quality['dff_max']) > 10,
            dff_implosion=np.abs(roi_quality['dff_min']) > 5,
            high_dff_noise=roi_quality['dff_std'] > 2
        )
    
    # Create composite quality flags
    roi_quality = add_roi_flags(
        roi_quality,
        exclude=roi_quality.get('dff_explosion', False) | 
                roi_quality.get('high_fc_negative', False),
        caution=roi_quality.get('high_alpha', False) | 
                roi_quality.get('low_snr', False) | 
                roi_quality.get('high_dff_noise', False),
        good_quality=(not roi_quality.get('exclude', True)) & 
                    (not roi_quality.get('caution', True))
    )
    
    # Store in data
    data['roi_quality'] = roi_quality
    
    return data

# Easy filtering functions
def get_roi_mask(data: Dict[str, Any], **conditions) -> np.ndarray:
    """Get boolean mask for ROIs based on quality conditions"""
    roi_df = data['roi_quality']
    mask = pd.Series(True, index=roi_df.index)
    
    for condition, value in conditions.items():
        if condition not in roi_df.columns:
            print(f"Warning: condition '{condition}' not found in ROI quality data")
            continue
        mask &= (roi_df[condition] == value)
    
    return mask.values

def filter_rois(data: Dict[str, Any], **conditions) -> Dict[str, Any]:
    """Return data with filtered ROIs"""
    mask = get_roi_mask(data, **conditions)
    roi_indices = np.where(mask)[0]
    
    filtered_data = {}
    for key, value in data.items():
        if key == 'roi_quality':
            filtered_data[key] = value.iloc[mask]
        elif isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == len(mask):
            # Filter first dimension if it matches number of ROIs
            filtered_data[key] = value[mask]
        else:
            # Keep unchanged (metadata, etc.)
            filtered_data[key] = value
    
    return filtered_data

# Usage examples:
def example_usage(data: Dict[str, Any]) -> None:
    """Example of how to use ROI quality tracking"""
    
    # Process and track quality
    data = process_rois_with_quality_tracking(data, cfg)
    
    # Look at quality summary
    roi_df = data['roi_quality']
    print(f"Total ROIs: {len(roi_df)}")
    print(f"Good quality: {roi_df['good_quality'].sum()}")
    print(f"Caution: {roi_df['caution'].sum()}")
    print(f"Exclude: {roi_df['exclude'].sum()}")
    
    # Filter for analysis
    good_rois = filter_rois(data, good_quality=True)
    print(f"Good ROIs for analysis: {good_rois['F'].shape[0]}")
    
    # Get specific subsets
    high_snr_mask = get_roi_mask(data, low_snr=False, exclude=False)
    analysis_data = data['dFF_clean'][high_snr_mask]
    
    # Save quality data
    roi_df.to_csv('roi_quality_metrics.csv')
    



































# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("=== SUITE2P PROCESSING PIPELINE ===")
    
    # Configuration
    cfg_path = r"D:/PHD/GIT/data_analysis/DAP/imaging/config.yaml"
    cfg = load_cfg_yaml(cfg_path)
    # validate_cfg(cfg)
    
    print("\n" + "="*50)
    print("STARTING PIPELINE")
    print("="*50)


# %%
# STEP 1: Load Suite2p data
data = load_suite2p(cfg)
data = subset_cells(data, cfg)


# %%
# STEP 2: Load imaging metadata
metadata = load_imaging_metadata(cfg)
data['imaging_metadata'] = metadata
print(f"Pixel size: {metadata.get('pixel_size_um', 0.5):.3f} µm/pixel")
# %%
# STEP 3: Scale-aware ROI characterization
select_anatomical_image(data, cfg)
extract_scale_aware_roi_features(data, cfg)

# %%
# STEP 4: Optional Cellpose integration
integrate_cellpose_iou(data, cfg)

# %%
# STEP 5: Classify with tie-breakers (will use Cellpose IoU data)
classify_rois_scale_aware(data, cfg)
print_label_summary(data)

# %%
# STEP 6: Generate overlay
generate_refined_label_overlay(data, cfg, show=True, save=True)
# %%

# Check what's happening with uncertain/process ROIs that look soma-like
feats = data['roi_features']
labels = np.array(data['roi_labels'])

# Find uncertain ROIs with soma-like sizes
unc_mask = labels == 'uncertain'
large_unc = unc_mask & (feats['area_um2'] > 80) & (feats['major_axis_um'] > 8)

print(f"Large uncertain ROIs: {large_unc.sum()}")
if large_unc.sum() > 0:
    print("Sample large uncertain ROIs:")
    for i in np.where(large_unc)[0][:]:
        print(f"  ROI {i}: area={feats['area_um2'][i]:.1f} µm², diameter={feats['major_axis_um'][i]:.1f} µm, AR={feats['aspect_ratio'][i]:.2f}, solidity={feats['solidity'][i]:.3f}, circ={feats['circularity'][i]:.3f}")


# %%


# Check that indexing is consistent
n_rois = data['F'].shape[0]
print(f"Consistent shapes:")
print(f"  F: {data['F'].shape}")
print(f"  stat: {len(data['stat'])}")
print(f"  labels: {len(data['roi_labels'])}")
print(f"  features area: {len(data['roi_features']['area_um2'])}")

# Map back to original Suite2p indices
if 'roi_mask' in data:
    original_indices = np.where(data['roi_mask'])[0]
    print(f"  Original Suite2p indices: {original_indices[:5]}... (showing first 5)")

# %%
print("\n" + "="*50)
print("CREATING INDEX MAPPING & SPATIAL DATA")
print("="*50)

# Create comprehensive ROI tracking
data = create_roi_index_mapping(data)


# %%
# Create spatial DataFrame for analysis
roi_df = create_spatial_roi_dataframe(data)
print(f"\nSpatial DataFrame preview:")
if roi_df is not None:
    print(roi_df[['filtered_idx', 'original_suite2p_idx', 'roi_label', 'area_um2', 'centroid_x_um', 'centroid_y_um']].head())


# %%
# Plot spatial distribution
plot_spatial_roi_distribution(data, cfg, color_by='roi_label', save=True)
plot_spatial_roi_distribution(data, cfg, color_by='area_um2', save=True)


# %%
# Export for external analysis
export_roi_spatial_data(data, cfg)






# %%
# STEP 1: Load Suite2p data
data = load_suite2p(cfg)
data = subset_cells(data, cfg)


# %%
# STEP 2: Load imaging metadata
metadata = load_imaging_metadata(cfg)
data['imaging_metadata'] = metadata
print(f"Pixel size: {metadata.get('pixel_size_um', 0.5):.3f} µm/pixel")
# %%
# STEP 3: Scale-aware ROI characterization
select_anatomical_image(data, cfg)
extract_scale_aware_roi_features(data, cfg)

# %%
# STEP 4: Optional Cellpose integration
integrate_cellpose_iou(data, cfg)

# %%
# STEP 5: Classify with tie-breakers (will use Cellpose IoU data)
classify_rois_scale_aware(data, cfg)
print_label_summary(data)

# %%
# STEP 6: Generate overlay
generate_refined_label_overlay(data, cfg, show=True, save=True)



# %%
# Create comprehensive ROI tracking
data = create_roi_index_mapping(data)


# %%
# Create spatial DataFrame for analysis
roi_df = create_spatial_roi_dataframe(data)
print(f"\nSpatial DataFrame preview:")
if roi_df is not None:
    print(roi_df[['filtered_idx', 'original_suite2p_idx', 'roi_label', 'area_um2', 'centroid_x_um', 'centroid_y_um']].head())


# %%
n_rois = data['F'].shape[0]
data['roi_quality'] = initialize_roi_quality_df(n_rois)


# %%
# Check if there are any bad frames
bad_frames = data['ops']['badframes']
has_bad_frames = np.any(bad_frames)
print(f"Has bad frames: {has_bad_frames}")

if has_bad_frames:
    num_bad_frames = np.sum(bad_frames)
    bad_frame_indices = np.where(bad_frames)[0]
    
    print(f"Number of bad frames: {num_bad_frames}")
    print(f"Bad frame indices: {bad_frame_indices}")
    print(f"Percentage of bad frames: {num_bad_frames/len(bad_frames)*100:.2f}%")
else:
    print("No bad frames detected!")






# %%

import matplotlib.pyplot as plt
import numpy as np

# Extract motion correction data from ops
ops = data['ops']
xoff = ops['xoff']
yoff = ops['yoff'] 
corrXY = ops['corrXY']


print(f"Motion correction data shapes:")
print(f"xoff: {xoff.shape}")
print(f"yoff: {yoff.shape}")
print(f"corrXY: {corrXY.shape}")

# Basic statistics
print(f"\nMotion statistics:")
print(f"X offset - mean: {np.mean(xoff):.2f}, std: {np.std(xoff):.2f}, range: [{np.min(xoff):.2f}, {np.max(xoff):.2f}]")
print(f"Y offset - mean: {np.mean(yoff):.2f}, std: {np.std(yoff):.2f}, range: [{np.min(yoff):.2f}, {np.max(yoff):.2f}]")
print(f"Correlation - mean: {np.mean(corrXY):.3f}, std: {np.std(corrXY):.3f}, range: [{np.min(corrXY):.3f}, {np.max(corrXY):.3f}]")

# Plot motion over time
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# X offset over time
axes[0].plot(xoff, 'b-', alpha=0.7)
axes[0].set_ylabel('X offset (pixels)')
axes[0].set_title('Motion Correction - X Offset')
axes[0].grid(True, alpha=0.3)

# Y offset over time  
axes[1].plot(yoff, 'r-', alpha=0.7)
axes[1].set_ylabel('Y offset (pixels)')
axes[1].set_title('Motion Correction - Y Offset')
axes[1].grid(True, alpha=0.3)

# Correlation over time
axes[2].plot(corrXY, 'g-', alpha=0.7)
axes[2].set_ylabel('Correlation')
axes[2].set_xlabel('Frame number')
axes[2].set_title('Motion Correction - Correlation')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Check for potential issues
print(f"\nPotential motion issues:")
large_x_motion = np.abs(xoff) > 10
large_y_motion = np.abs(yoff) > 10
low_correlation = corrXY < 0.5

print(f"Frames with large X motion (>10 pixels): {np.sum(large_x_motion)} ({100*np.sum(large_x_motion)/len(xoff):.1f}%)")
print(f"Frames with large Y motion (>10 pixels): {np.sum(large_y_motion)} ({100*np.sum(large_y_motion)/len(yoff):.1f}%)")
print(f"Frames with low correlation (<0.5): {np.sum(low_correlation)} ({100*np.sum(low_correlation)/len(corrXY):.1f}%)")

# %%


F = data['F']          # shape: (n_rois, n_frames)
Fneu = data['Fneu']    # shape: (n_rois, n_frames)
n_rois, n_frames = F.shape
eps = 1e-9

# Basic moments / percentiles
F_mean     = np.mean(F, axis=1)
F_std      = np.std(F, axis=1)
F_median   = np.median(F, axis=1)
F_min      = np.min(F, axis=1)
F_max      = np.max(F, axis=1)
F_p05      = np.percentile(F, 5, axis=1)
F_p95      = np.percentile(F, 95, axis=1)
F_range    = F_max - F_min
F_cv       = F_std / (F_mean + eps)

# Robust counts
F_zero_pct = np.mean(F == 0, axis=1) * 100.0
F_neg_pct  = np.mean(F < 0, axis=1) * 100.0
F_nan_pct  = np.mean(np.isnan(F), axis=1) * 100.0

# Same for neuropil
Fneu_mean   = np.mean(Fneu, axis=1)
Fneu_std    = np.std(Fneu, axis=1)
Fneu_median = np.median(Fneu, axis=1)
Fneu_min    = np.min(Fneu, axis=1)
Fneu_max    = np.max(Fneu, axis=1)
Fneu_p05    = np.percentile(Fneu, 5, axis=1)
Fneu_p95    = np.percentile(Fneu, 95, axis=1)
Fneu_range  = Fneu_max - Fneu_min
Fneu_cv     = Fneu_std / (Fneu_mean + eps)
Fneu_neg_pct = np.mean(Fneu < 0, axis=1) * 100.0
Fneu_nan_pct = np.mean(np.isnan(Fneu), axis=1) * 100.0

# Relationship metrics
# per-ROI Pearson r between F and Fneu (safe when variance > 0)
F_Fneu_corr = np.full(n_rois, np.nan)
for i in range(n_rois):
    fx = F[i]
    fy = Fneu[i]
    if np.nanstd(fx) > 0 and np.nanstd(fy) > 0:
        F_Fneu_corr[i] = np.corrcoef(np.nan_to_num(fx), np.nan_to_num(fy))[0,1]

# fraction of frames where neuropil exceeds cell signal
Fneu_gt_F_pct = np.mean(Fneu > F, axis=1) * 100.0

# Put into roi_quality DataFrame
roi_df = data.get('roi_quality')
if roi_df is None:
    roi_df = initialize_roi_quality_df(n_rois)

# Add/update columns (use clear logical names)
roi_df['F_mean']        = F_mean
roi_df['F_std']         = F_std
roi_df['F_median']      = F_median
roi_df['F_min']         = F_min
roi_df['F_max']         = F_max
roi_df['F_p05']         = F_p05
roi_df['F_p95']         = F_p95
roi_df['F_range']       = F_range
roi_df['F_cv']          = F_cv
roi_df['F_zero_pct']    = F_zero_pct
roi_df['F_neg_pct']     = F_neg_pct
roi_df['F_nan_pct']     = F_nan_pct

roi_df['Fneu_mean']     = Fneu_mean
roi_df['Fneu_std']      = Fneu_std
roi_df['Fneu_median']   = Fneu_median
roi_df['Fneu_min']      = Fneu_min
roi_df['Fneu_max']      = Fneu_max
roi_df['Fneu_p05']      = Fneu_p05
roi_df['Fneu_p95']      = Fneu_p95
roi_df['Fneu_range']    = Fneu_range
roi_df['Fneu_cv']       = Fneu_cv
roi_df['Fneu_neg_pct']  = Fneu_neg_pct
roi_df['Fneu_nan_pct']  = Fneu_nan_pct

roi_df['F_Fneu_corr']   = F_Fneu_corr
roi_df['Fneu_gt_F_pct'] = Fneu_gt_F_pct

data['roi_quality'] = roi_df

# Example prints (use stored vars if you want to log summary)
n_neg_rois = np.sum(F_neg_pct > 0)
n_high_fneu_gt = np.sum(Fneu_gt_F_pct > 10.0)  # example threshold
print(f"F shape: {F.shape}; n_rois with any negative F: {n_neg_rois}; rois with >10% frames Fneu>F: {n_high_fneu_gt}")

# %%

F = data['F']

baseline_size=1001
baseline_pct=20
resid_size=301
resid_pct=50
signal_pct=95

# 1. baseline detrend with long running percentile
F0 = percentile_filter(F, size=baseline_size, percentile=baseline_pct)
# avoid divide-by-zero
eps = np.finfo(float).eps
F0 = np.where(F0 <= 0, eps, F0)
dff = (F - F0) / F0

# 2. noise = MAD of high-pass residual
resid = dff - percentile_filter(dff, size=resid_size, percentile=resid_pct)
mad = np.median(np.abs(resid - np.median(resid)))
noise = 1.4826 * mad

# 3. signal = high percentile of dF/F (robust peak estimate)
signal = np.percentile(dff, signal_pct)

F_snr = signal / (noise + eps)





# %%
# STEP 7: Fluorescence processing
data = neuropil_regress(data, cfg)


# %%

# Add these diagnostic lines after the neuropil correction is complete:

print(f"\n=== NEUROPIL CORRECTION DIAGNOSTICS ===")


F = data['F']
Fc = data['Fc']
N = F.shape[0]



# Check for negative or near-zero baseline values
negative_baseline = np.sum(Fc <= 0, axis=1)  # Count negative/zero values per ROI
roi_with_negatives = np.sum(negative_baseline > 0)
print(f"ROIs with negative/zero values: {roi_with_negatives}/{N}")
if roi_with_negatives > 0:
    print(f"  Max negative values in single ROI: {negative_baseline.max()}")

# Check baseline levels (mean of corrected signal)
baseline_levels = np.mean(Fc, axis=1)
low_baseline_rois = np.sum(baseline_levels < 50)  # Adjust threshold as needed
print(f"ROIs with low baseline (<50): {low_baseline_rois}/{N}")
print(f"Baseline range: [{baseline_levels.min():.1f}, {baseline_levels.max():.1f}]")

# Check for signal inversion (corrected signal lower than original)
signal_reduction = np.mean(F - Fc, axis=1) / np.mean(F, axis=1)
inverted_rois = np.sum(signal_reduction > 0.5)  # More than 50% signal removed
print(f"ROIs with >50% signal reduction: {inverted_rois}/{N}")
if inverted_rois > 0:
    print(f"  Max signal reduction: {signal_reduction.max():.2f}")

# Check for extreme correction values
correction_magnitude = np.mean(np.abs(F-Fc), axis=1)
original_magnitude = np.mean(F, axis=1)
overcorrection_ratio = correction_magnitude / original_magnitude
extreme_correction = np.sum(overcorrection_ratio > 0.8)
print(f"ROIs with correction >80% of signal: {extreme_correction}/{N}")

# Check variance changes (important for dF/F stability)
var_ratio = np.var(Fc, axis=1) / np.var(F, axis=1)
high_var_rois = np.sum(var_ratio > 2.0)  # Variance doubled
print(f"ROIs with >2x variance increase: {high_var_rois}/{N}")

# Flag problematic ROIs for potential exclusion
problematic_mask = (
    (negative_baseline > T * 0.1) |  # >10% negative values
    (baseline_levels < 20) |         # Very low baseline
    (signal_reduction > 0.7) |       # >70% signal removed
    (var_ratio > 3.0)               # >3x variance increase
)
problematic_count = np.sum(problematic_mask)
print(f"\nPotentially problematic ROIs: {problematic_count}/{N}")

if problematic_count > 0:
    print("Consider excluding these ROIs or using fallback correction")
    data["problematic_rois"] = problematic_mask



# %%

# example_usage(data)




# %%
# DIAGNOSE NEUROPIL CORRECTION ISSUES
print("\n=== DIAGNOSING NEUROPIL CORRECTION ===")

# Check for problematic neuropil corrections
problem_rois_neuropil = []
for roi in range(data['F'].shape[0]):
    # Check if Fc goes negative extensively
    negative_fraction = np.sum(data['Fc'][roi] < 0) / len(data['Fc'][roi])
    if negative_fraction > 0.1:  # >10% negative
        problem_rois_neuropil.append(('excessive_negative', roi, negative_fraction))
    
    # Check if correction inverts the signal
    f_mean = np.mean(data['F'][roi])
    fc_mean = np.mean(data['Fc'][roi])
    if fc_mean < 0 and f_mean > 0:
        problem_rois_neuropil.append(('signal_inversion', roi, fc_mean))
    
    # Check for extreme alpha values
    if 'neuropil_a' in data:
        alpha = data['neuropil_a'][roi]
        if alpha > 0.6:  # Very high neuropil subtraction
            problem_rois_neuropil.append(('high_alpha', roi, alpha))

print(f"Neuropil correction diagnostics:")
print(f"  Total ROIs: {data['F'].shape[0]}")
print(f"  Problem ROIs found: {len(problem_rois_neuropil)}")

if len(problem_rois_neuropil) > 0:
    print(f"  Problem breakdown:")
    problem_types = {}
    for prob_type, roi, value in problem_rois_neuropil:
        if prob_type not in problem_types:
            problem_types[prob_type] = []
        problem_types[prob_type].append((roi, value))
    
    for prob_type, cases in problem_types.items():
        print(f"    {prob_type}: {len(cases)} ROIs")
        # Show worst 3 cases
        if prob_type == 'excessive_negative':
            worst = sorted(cases, key=lambda x: x[1], reverse=True)[:3]
            for roi, frac in worst:
                print(f"      ROI {roi}: {100*frac:.1f}% negative")
        elif prob_type == 'signal_inversion':
            for roi, fc_mean in cases[:3]:
                print(f"      ROI {roi}: Fc mean = {fc_mean:.1f}")
        elif prob_type == 'high_alpha':
            worst = sorted(cases, key=lambda x: x[1], reverse=True)[:3]
            for roi, alpha in worst:
                print(f"      ROI {roi}: alpha = {alpha:.3f}")

# Basic Fc statistics
print(f"\nNeuropil-corrected fluorescence (Fc) statistics:")
print(f"  Range: {np.min(data['Fc']):.1f} to {np.max(data['Fc']):.1f}")
print(f"  Mean: {np.mean(data['Fc']):.1f}")
print(f"  Std: {np.std(data['Fc']):.1f}")

if 'neuropil_a' in data:
    print(f"  Alpha coefficients: {np.mean(data['neuropil_a']):.3f} ± {np.std(data['neuropil_a']):.3f}")
    print(f"    Range: {np.min(data['neuropil_a']):.3f} to {np.max(data['neuropil_a']):.3f}")





# %%
# Add this to your main analysis section after the dF/F diagnosis
diagnose_problematic_roi_signals(data, cfg)



# %%

# With this:
diagnose_problematic_roi_signals_comprehensive(data, cfg)




# %%
# COMPREHENSIVE ROI DIAGNOSTICS
run_comprehensive_diagnostics(data, cfg)

# %%
# For quick testing of specific ROIs
roi_to_check = 2  # Change this to any ROI you want to examine
fig = plot_comprehensive_roi_diagnostic(data, cfg, roi_to_check, save=False)
plt.show()



# %%
# For quick testing of specific ROIs
roi_to_check = 2  # Change this to any ROI you want to examine
for roi_to_check in range(30):
    fig = plot_comprehensive_roi_diagnostic(data, cfg, roi_to_check, save=False)
    plt.show()




# %%


roi = 12
print(f"ROI {roi} stored data check:")
print(f"  F range: [{data['F'][roi].min():.1f}, {data['F'][roi].max():.1f}]")
print(f"  Fneu range: [{data['Fneu'][roi].min():.1f}, {data['Fneu'][roi].max():.1f}]") 
print(f"  Fc range: [{data['Fc'][roi].min():.1f}, {data['Fc'][roi].max():.1f}]")
print(f"  F0 range: [{data['F0'][roi].min():.1f}, {data['F0'][roi].max():.1f}]")
print(f"  dFF range: [{data['dFF_clean'][roi].min():.1f}, {data['dFF_clean'][roi].max():.1f}]")

# Check a specific time segment that shows problems in review plot
segment_start = int(0.5 * 30)  # Around 0.5 seconds at 30 Hz
segment_end = int(40 * 30)     # Around 40 seconds
print(f"\nSegment check (frames {segment_start}:{segment_end}):")
print(f"  Fc segment: [{data['Fc'][roi, segment_start:segment_end].min():.1f}, {data['Fc'][roi, segment_start:segment_end].max():.1f}]")
print(f"  F0 segment: [{data['F0'][roi, segment_start:segment_end].min():.1f}, {data['F0'][roi, segment_start:segment_end].max():.1f}]")



# %%

# Check the worst ROIs from your comprehensive diagnosis
worst_rois = [2, 18, 21]  # These were showing major issues

for roi in worst_rois:
    print(f"\nROI {roi} diagnostic:")
    print(f"  F range: [{data['F'][roi].min():.1f}, {data['F'][roi].max():.1f}]")
    print(f"  Fneu range: [{data['Fneu'][roi].min():.1f}, {data['Fneu'][roi].max():.1f}]") 
    print(f"  Fc range: [{data['Fc'][roi].min():.1f}, {data['Fc'][roi].max():.1f}]")
    print(f"  F0 range: [{data['F0'][roi].min():.1f}, {data['F0'][roi].max():.1f}]")
    print(f"  dFF range: [{data['dFF_clean'][roi].min():.1f}, {data['dFF_clean'][roi].max():.1f}]")
    
    # Check for the classic problematic patterns:
    fc = data['Fc'][roi]
    f0 = data['F0'][roi]
    
    # 1. Fc dropping below F0 significantly
    negative_dff_mask = fc < f0 * 0.5  # Fc less than 50% of F0
    severe_negative = np.sum(negative_dff_mask)
    
    # 2. F0 near zero
    near_zero_f0 = np.sum(f0 < 0.1)
    
    # 3. F0 much larger than typical Fc
    f0_too_high = np.sum(f0 > fc * 2)  # F0 more than 2x Fc
    
    print(f"  Problematic patterns:")
    print(f"    Fc << F0 (severe negative): {severe_negative}/{len(fc)} frames ({100*severe_negative/len(fc):.1f}%)")
    print(f"    F0 near zero: {near_zero_f0}/{len(f0)} frames ({100*near_zero_f0/len(f0):.1f}%)")
    print(f"    F0 >> Fc: {f0_too_high}/{len(f0)} frames ({100*f0_too_high/len(f0):.1f}%)")


# %%


# Check if you have any truly problematic ROIs
truly_bad_rois = []
for roi in range(data['dFF_clean'].shape[0]):
    dff = data['dFF_clean'][roi]
    if np.max(dff) > 10 or np.min(dff) < -5:
        truly_bad_rois.append(roi)
        
print(f"Truly problematic ROIs: {truly_bad_rois}")




# %%
cfg = load_cfg_yaml(cfg_path)
data = compute_baseline_and_dff(data, cfg)










# %%

import umap
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get the traces data see
traces = data['dFF_clean']  # shape should be (n_rois, n_timepoints)
print(f"Traces shape: {traces.shape}")

# Normalize traces (z-score each ROI)
traces_norm = (traces - traces.mean(1, keepdims=True)) / traces.std(1, keepdims=True)

# Remove any NaN values that might cause issues
mask = ~np.isnan(traces_norm).any(axis=1)
traces_clean = traces_norm[mask]
print(f"Clean traces shape: {traces_clean.shape}")

# Reduce to 10D latent space with UMAP
print("Running UMAP dimensionality reduction...")
embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=10, random_state=42).fit_transform(traces_clean)

# Cluster in latent space
print("Running HDBSCAN clustering...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
clusters = clusterer.fit_predict(embedding)

# Print clustering results
unique_clusters = np.unique(clusters)
print(f"Found {len(unique_clusters)} clusters (including noise = -1)")
for cluster_id in unique_clusters:
    count = np.sum(clusters == cluster_id)
    if cluster_id == -1:
        print(f"  Noise: {count} ROIs")
    else:
        print(f"  Cluster {cluster_id}: {count} ROIs")



# %%





import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import median_abs_deviation
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan

X = data['dFF_clean']  # (n_rois, n_time)

# 1) Robust z-score per trace
med = np.nanmedian(X, axis=1, keepdims=True)
mad = median_abs_deviation(X, axis=1, scale='normal')[:, None]
mad[mad == 0] = 1.0
Xr = (X - med) / mad

# Clip extreme outliers (protects distances)
Xr = np.clip(Xr, -5, 5)

# Remove NaN rows (and constant rows)
row_ok = ~np.isnan(Xr).any(axis=1) & (Xr.std(1) > 1e-8)
Xr = Xr[row_ok]
n_rois = Xr.shape[0]

# 2) Light temporal coarsening (mean-pool by factor q)
# q = 2  # try 2 or 3 if you have lots of frames
# if q > 1:
#     T = (Xr.shape[1] // q) * q
#     Xr = Xr[:, :T].reshape(n_rois, T//q, q).mean(axis=2)

# 3) PCA (denoise + speed)
pca = PCA(n_components=min(50, Xr.shape[1]), svd_solver='auto', random_state=42)
Xp = pca.fit_transform(Xr)

# Optional: whiten PCA space for UMAP stability
Xp = StandardScaler(with_mean=False, with_std=True).fit_transform(Xp)

# 4) UMAP on PCA space (shape-aware metric)
um = umap.UMAP(
    n_neighbors=40,
    min_dist=0.05,
    n_components=10,              # 5–10 if you’ll cluster in latent space
    metric='cosine',              # or 'correlation'
    random_state=42,
    densmap=False
)
Z = um.fit_transform(Xp)

# 5) HDBSCAN in latent space
min_cluster_size = max(30, int(0.03 * n_rois))
min_samples = max(10, min_cluster_size // 2)

hdb = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    cluster_selection_method='eom',
    prediction_data=False
)
clusters = hdb.fit_predict(Z)
probs = hdb.probabilities_  # cluster membership strength

# Example: confidence mask for "keep" clusters
conf_keep = (probs >= 0.6) & (clusters >= 0)

# Print clustering results
unique_clusters = np.unique(clusters)
print(f"Found {len(unique_clusters)} clusters (including noise = -1)")
for cluster_id in unique_clusters:
    count = np.sum(clusters == cluster_id)
    if cluster_id == -1:
        print(f"  Noise: {count} ROIs")
    else:
        print(f"  Cluster {cluster_id}: {count} ROIs")













# %%

# Visualize the clustering results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 2D UMAP for visualization
embedding_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit_transform(traces_clean)

# Plot 2D embedding colored by clusters
scatter = axes[0,0].scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=clusters, cmap='tab10', alpha=0.7, s=20)
axes[0,0].set_title('2D UMAP Embedding (colored by clusters)')
axes[0,0].set_xlabel('UMAP 1')
axes[0,0].set_ylabel('UMAP 2')

# 2. Cluster sizes
cluster_counts = []
cluster_labels = []
for cluster_id in unique_clusters:
    count = np.sum(clusters == cluster_id)
    cluster_counts.append(count)
    if cluster_id == -1:
        cluster_labels.append('Noise')
    else:
        cluster_labels.append(f'C{cluster_id}')

axes[0,1].bar(range(len(cluster_counts)), cluster_counts)
axes[0,1].set_xticks(range(len(cluster_counts)))
axes[0,1].set_xticklabels(cluster_labels, rotation=45)
axes[0,1].set_title('Cluster Sizes')
axes[0,1].set_ylabel('Number of ROIs')

# 3. Average traces for each cluster
axes[1,0].set_title('Average Traces by Cluster')
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

for i, cluster_id in enumerate(unique_clusters):
    if cluster_id == -1:  # Skip noise for now
        continue
    cluster_mask = clusters == cluster_id
    cluster_traces = traces_clean[cluster_mask]
    mean_trace = cluster_traces.mean(axis=0)
    
    axes[1,0].plot(mean_trace, color=colors[i], label=f'Cluster {cluster_id} (n={np.sum(cluster_mask)})', linewidth=2)

axes[1,0].set_xlabel('Time')
axes[1,0].set_ylabel('Normalized ΔF/F')
axes[1,0].legend()

# 4. Heatmap of sample traces from each cluster
axes[1,1].set_title('Sample Traces from Each Cluster')
plot_traces = []
plot_labels = []

for cluster_id in unique_clusters:
    if cluster_id == -1:  # Skip noise
        continue
    cluster_mask = clusters == cluster_id
    cluster_traces = traces_clean[cluster_mask]
    
    # Sample up to 5 traces from each cluster
    n_samples = min(5, cluster_traces.shape[0])
    sample_indices = np.random.choice(cluster_traces.shape[0], n_samples, replace=False)
    
    for idx in sample_indices:
        plot_traces.append(cluster_traces[idx])
        plot_labels.append(f'C{cluster_id}')

if plot_traces:
    im = axes[1,1].imshow(plot_traces, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('Sample ROIs')
    plt.colorbar(im, ax=axes[1,1])

plt.tight_layout()
plt.show()




# %%



# Let's also look at cluster quality metrics
print(f"HDBSCAN cluster probabilities range: {clusterer.probabilities_.min():.3f} - {clusterer.probabilities_.max():.3f}")
print(f"Number of high-confidence assignments (prob > 0.5): {np.sum(clusterer.probabilities_ > 0.5)}")

# Plot probability distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(clusterer.probabilities_, bins=50, alpha=0.7)
plt.xlabel('Cluster Assignment Probability')
plt.ylabel('Count')
plt.title('Distribution of Cluster Probabilities')

plt.subplot(1, 2, 2)
# Show probabilities colored by cluster
scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=clusterer.probabilities_, cmap='viridis', alpha=0.7, s=20)
plt.colorbar(scatter, label='Cluster Probability')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('Cluster Assignment Confidence')

plt.tight_layout()
plt.show()



# %%



# Analyze temporal patterns within clusters
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

valid_clusters = [c for c in unique_clusters if c != -1]

for i, cluster_id in enumerate(valid_clusters[:6]):  # Show up to 6 clusters
    if i >= 6:
        break
        
    cluster_mask = clusters == cluster_id
    cluster_traces = traces_clean[cluster_mask]
    
    # Plot individual traces (sample) + mean
    n_plot = min(20, cluster_traces.shape[0])
    sample_indices = np.random.choice(cluster_traces.shape[0], n_plot, replace=False)
    
    for j in sample_indices:
        axes[i].plot(cluster_traces[j], alpha=0.3, color='lightblue', linewidth=0.5)
    
    # Plot mean trace
    mean_trace = cluster_traces.mean(axis=0)
    std_trace = cluster_traces.std(axis=0)
    
    axes[i].plot(mean_trace, color='red', linewidth=2, label='Mean')
    axes[i].fill_between(range(len(mean_trace)), 
                        mean_trace - std_trace, 
                        mean_trace + std_trace, 
                        alpha=0.2, color='red', label='±1 SD')
    
    axes[i].set_title(f'Cluster {cluster_id} (n={np.sum(cluster_mask)})')
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Normalized ΔF/F')
    axes[i].legend()

# Hide unused subplots
for i in range(len(valid_clusters), 6):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()









# %%





import matplotlib.pyplot as plt
import numpy as np

def plot_cluster_traces(data, clusters, cluster_labels, start_time=0, n_samples=5, duration=10):
    """
    Plot sample traces from each cluster in separate figures, with each trace as its own subplot
    
    Parameters:
    -----------
    data : dict
        Contains 'dFF_clean' (n_rois x n_timepoints) and 'imaging_time' (timepoints,)
    clusters : array
        Length n_rois, indicates which cluster each ROI belongs to
    cluster_labels : dict
        Maps cluster IDs to their string labels
    start_time : float
        Start time in seconds for the segment to plot
    n_samples : int
        Number of sample traces to plot per cluster
    duration : float
        Duration in seconds of segment to plot
    """
    
    # Get time segment
    time = data['imaging_time']
    end_time = start_time + duration
    time_mask = (time >= start_time) & (time <= end_time)
    time_segment = time[time_mask]
    
    # Get unique cluster IDs
    unique_clusters = np.unique(clusters)
    
    # Create separate figure for each cluster
    for cluster_id in unique_clusters:
        # Get ROIs belonging to this cluster
        cluster_mask = clusters == cluster_id
        cluster_rois = np.where(cluster_mask)[0]
        
        # Select random sample of ROIs from this cluster
        n_rois_in_cluster = len(cluster_rois)
        sample_size = min(n_samples, n_rois_in_cluster)
        sampled_rois = np.random.choice(cluster_rois, size=sample_size, replace=False)
        
        # Create figure with subplots for each trace
        fig, axes = plt.subplots(sample_size, 1, figsize=(12, 2*sample_size), sharex=True)
        if sample_size == 1:
            axes = [axes]
        
        # Get cluster label
        try:
            cluster_label = cluster_labels[cluster_id] if cluster_id < len(cluster_labels) else f'Cluster {cluster_id}'
        except (IndexError, TypeError):
            cluster_label = f'Cluster {cluster_id}'
        
        # Plot each trace in its own subplot
        for i, roi_idx in enumerate(sampled_rois):
            trace_segment = data['dFF_clean'][roi_idx, time_mask]
            axes[i].plot(time_segment, trace_segment, linewidth=1.5, color='blue')
            axes[i].set_ylabel('dF/F')
            axes[i].set_title(f'ROI {roi_idx}')
            axes[i].grid(True, alpha=0.3)
        
        # Set overall figure title and x-label
        fig.suptitle(f'{cluster_label} (n={n_rois_in_cluster}, showing {sample_size} traces)', 
                     fontsize=14, y=0.98)
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for suptitle
        plt.show()

# Example usage:
plot_cluster_traces(data, clusters, cluster_labels, start_time=50, n_samples=20, duration=10)




















# %%

import numpy as np
from scipy.stats import median_abs_deviation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan



from scipy.signal import correlate
from numpy.fft import rfft, rfftfreq

def _autocorr_feats(X, max_lag=100):
    # X: (n_rois, T)
    Xz = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    n, T = Xz.shape
    feats = np.empty((n, max_lag), dtype=np.float32)
    for i in range(n):
        ac = correlate(Xz[i], Xz[i], mode="full")
        mid = len(ac)//2
        seg = ac[mid:mid+max_lag]
        # normalize by ac[0] to avoid amplitude bias
        feats[i] = seg / (seg[0] + 1e-8)
    return feats

def _fft_feats(X, fs=None, keep=128):
    # magnitude spectrum (power); drop DC, log-compress
    Xz = (X - X.mean(1, keepdims=True))
    spec = np.abs(rfft(Xz, axis=1))
    spec = spec[:, 1:keep+1]    # drop DC, limit bins
    spec = np.log1p(spec)       # compress spikes
    # per-row normalize
    spec = (spec - spec.mean(1, keepdims=True)) / (spec.std(1, keepdims=True)+1e-8)
    return spec.astype(np.float32)

def _diff_feats(X, k=1):
    # k=1 → first difference; emphasizes slopes/decays
    D = np.diff(X, n=k, axis=1)
    D = (D - D.mean(1, keepdims=True)) / (D.std(1, keepdims=True)+1e-8)
    return D.astype(np.float32)

def _maybe_circ_shift(X, enable=False, rng=None):
    if not enable: return X
    if rng is None: rng = np.random.default_rng(42)
    n, T = X.shape
    out = np.empty_like(X)
    for i in range(n):
        s = int(rng.integers(0, T))
        out[i] = np.roll(X[i], s)
    return out




def hierarchical_clustering(data, max_iterations=3, min_cluster_size_ratio=0.03, 
                          min_subcluster_size=15, confidence_threshold=0.6):
    """
    Perform hierarchical clustering with subclustering iterations.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing 'dFF_clean' key with shape (n_rois, n_time)
    max_iterations : int
        Maximum number of subclustering iterations
    min_cluster_size_ratio : float
        Minimum cluster size as ratio of total ROIs for initial clustering
    min_subcluster_size : int
        Minimum absolute size for subclusters
    confidence_threshold : float
        Minimum probability threshold for cluster membership
        
    Returns:
    --------
    results : dict
        Contains cluster hierarchy and ROI mappings
    """
    
    X = data['dFF_clean']  # (n_rois, n_time)
    original_n_rois = X.shape[0]
    
    # Keep track of original indices
    original_indices = np.arange(original_n_rois)
    
    # Store results for each iteration
    results = {
        'hierarchy': {},  # iteration -> cluster_id -> roi_indices
        'cluster_info': {},  # iteration -> cluster_id -> metadata
        'roi_to_cluster': {},  # iteration -> roi_index -> cluster_path
        'filtered_indices': None  # ROIs that passed initial filtering
    }
    
    def preprocess_data(X_input, roi_indices):
        """Preprocess data and return valid ROI mask"""
        # 1) Robust z-score per trace
        med = np.nanmedian(X_input, axis=1, keepdims=True)
        mad = median_abs_deviation(X_input, axis=1, scale='normal')[:, None]
        mad[mad == 0] = 1.0
        Xr = (X_input - med) / mad
        
        # Clip extreme outliers
        Xr = np.clip(Xr, -5, 5)
        
        # Remove NaN rows and constant rows
        row_ok = ~np.isnan(Xr).any(axis=1) & (Xr.std(1) > 1e-8)
        
        return Xr[row_ok], roi_indices[row_ok], row_ok
    
    def cluster_data(Xr, min_cluster_size, feature_mode="autocorr",
                    circ_shift=True, umap_metric='cosine'):
        """
        feature_mode: 'raw' | 'autocorr' | 'fft' | 'diff' | 'combo'
        circ_shift: random circular shift before featureizing (kills timing)
        """
        # --- shape-invariant featureization ---
        Xf_in = _maybe_circ_shift(Xr, enable=circ_shift)

        if feature_mode == "raw":
            Xf = Xf_in
        elif feature_mode == "autocorr":
            Xf = _autocorr_feats(Xf_in, max_lag=min(150, Xf_in.shape[1]//2))
        elif feature_mode == "fft":
            Xf = _fft_feats(Xf_in, keep=min(256, Xf_in.shape[1]//2))
        elif feature_mode == "diff":
            Xf = _diff_feats(Xf_in, k=1)
        elif feature_mode == "combo":
            # concatenate autocorr + fft (works very well)
            A = _autocorr_feats(Xf_in, max_lag=min(120, Xf_in.shape[1]//2))
            F = _fft_feats(Xf_in, keep=min(128, Xf_in.shape[1]//2))
            Xf = np.concatenate([A, F], axis=1)
        else:
            raise ValueError("Unknown feature_mode")

        n_rois = Xf.shape[0]

        # --- PCA (denoise/speed) ---
        pca = PCA(n_components=min(50, Xf.shape[0]), random_state=42)
        Xp = pca.fit_transform(Xf)
        Xp = StandardScaler(with_mean=False, with_std=True).fit_transform(Xp)

        # --- UMAP ---
        um = umap.UMAP(
            n_neighbors=min(40, max(10, n_rois // 3)),
            min_dist=0.05,
            n_components=min(10, max(2, n_rois // 2)),
            metric=umap_metric,         # 'cosine' or 'euclidean' both fine after featureization
            random_state=42
        )
        Z = um.fit_transform(Xp)

        # --- HDBSCAN ---
        min_samples = max(5, min_cluster_size // 2)
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method='eom',
            prediction_data=True
        )
        clusters = hdb.fit_predict(Z)
        probs = hdb.probabilities_

        return clusters, probs, pca, um, hdb
    
    
    # def cluster_data(Xr, min_cluster_size):
    #     """Perform PCA + UMAP + HDBSCAN clustering"""
    #     n_rois = Xr.shape[0]
        
    #     # 3) PCA
    #     pca = PCA(n_components=min(50, Xr.shape[1]), svd_solver='auto', random_state=42)
    #     Xp = pca.fit_transform(Xr)
        
    #     # Whiten PCA space
    #     Xp = StandardScaler(with_mean=False, with_std=True).fit_transform(Xp)
        
    #     # 4) UMAP
    #     um = umap.UMAP(
    #         n_neighbors=min(40, max(5, n_rois // 3)),  # Adaptive n_neighbors
    #         min_dist=0.05,
    #         n_components=min(10, n_rois // 2),
    #         metric='cosine',
    #         random_state=42,
    #         densmap=False
    #     )
    #     Z = um.fit_transform(Xp)
        
    #     # 5) HDBSCAN
    #     min_samples = max(5, min_cluster_size // 2)
        
    #     hdb = hdbscan.HDBSCAN(
    #         min_cluster_size=min_cluster_size,
    #         min_samples=min_samples,
    #         cluster_selection_method='eom',
    #         prediction_data=False
    #     )
    #     clusters = hdb.fit_predict(Z)
    #     probs = hdb.probabilities_
        
    #     return clusters, probs, pca, um, hdb
    
    # Queue for iterative processing: (iteration, parent_cluster_id, roi_indices, data_subset)
    processing_queue = [(0, 'root', original_indices, X)]
    
    while processing_queue and len([item for item in processing_queue if item[0] < max_iterations]) > 0:
        iteration, parent_cluster, roi_indices, X_subset = processing_queue.pop(0)
        
        if iteration >= max_iterations:
            continue
            
        print(f"\n=== Iteration {iteration}, Parent: {parent_cluster} ===")
        print(f"Processing {len(roi_indices)} ROIs")
        
        # Preprocess current data subset
        Xr, valid_roi_indices, row_ok = preprocess_data(X_subset, roi_indices)
        
        if iteration == 0:
            results['filtered_indices'] = valid_roi_indices
        
        n_valid_rois = Xr.shape[0]
        if n_valid_rois < 10:  # Skip if too few ROIs
            print(f"Skipping: only {n_valid_rois} valid ROIs")
            continue
        
        # Determine minimum cluster size
        if iteration == 0:
            min_cluster_size = max(30, int(min_cluster_size_ratio * n_valid_rois))
        else:
            min_cluster_size = min_subcluster_size
        
        # Perform clustering
        # clusters, probs, pca, um, hdb = cluster_data(Xr, min_cluster_size)
        
        # initial pass
        clusters, probs, pca, um, hdb = cluster_data(
            Xr, min_cluster_size,
            feature_mode="raw",      # <-- key change
            circ_shift=True,
            umap_metric='cosine'
        )        
        
        # Apply confidence filtering
        conf_keep = (probs >= confidence_threshold) & (clusters >= 0)
        
        # Store results for this iteration
        if iteration not in results['hierarchy']:
            results['hierarchy'][iteration] = {}
            results['cluster_info'][iteration] = {}
            results['roi_to_cluster'][iteration] = {}
        
        # Process each cluster
        unique_clusters = np.unique(clusters)
        print(f"Found {len(unique_clusters)} clusters (including noise = -1)")
        
        for cluster_id in unique_clusters:
            cluster_mask = (clusters == cluster_id)
            cluster_roi_indices = valid_roi_indices[cluster_mask]
            cluster_probs = probs[cluster_mask]
            
            count = np.sum(cluster_mask)
            
            if cluster_id == -1:
                print(f"  Noise: {count} ROIs")
                cluster_key = f"noise"
            else:
                print(f"  Cluster {cluster_id}: {count} ROIs")
                cluster_key = f"{parent_cluster}.{cluster_id}" if parent_cluster != 'root' else str(cluster_id)
                
                # Add to processing queue for next iteration if cluster is large enough
                if count >= min_subcluster_size * 2:  # Only subcluster if we can potentially split
                    cluster_data_subset = X[cluster_roi_indices]
                    processing_queue.append((iteration + 1, cluster_key, cluster_roi_indices, cluster_data_subset))
            
            # Store cluster information
            results['hierarchy'][iteration][cluster_key] = cluster_roi_indices
            results['cluster_info'][iteration][cluster_key] = {
                'size': count,
                'probabilities': cluster_probs,
                'parent': parent_cluster,
                'confidence_kept': np.sum(conf_keep[cluster_mask])
            }
            
            # Update ROI to cluster mapping
            for roi_idx in cluster_roi_indices:
                if roi_idx not in results['roi_to_cluster'][iteration]:
                    results['roi_to_cluster'][iteration][roi_idx] = cluster_key
    
    return results

# Usage example:
def print_hierarchy(results):
    """Print the clustering hierarchy in a readable format"""
    print("\n=== CLUSTERING HIERARCHY ===")
    
    for iteration in sorted(results['hierarchy'].keys()):
        print(f"\nIteration {iteration}:")
        for cluster_key, roi_indices in results['hierarchy'][iteration].items():
            info = results['cluster_info'][iteration][cluster_key]
            print(f"  {cluster_key}: {info['size']} ROIs (confidence kept: {info['confidence_kept']})")
            print(f"    ROI indices: {roi_indices[:10]}{'...' if len(roi_indices) > 10 else ''}")

# def get_roi_full_path(results, roi_index, target_iteration=None):
#     """Get the full cluster path for a specific ROI"""
#     if target_iteration is None:
#         target_iteration = max(results['roi_to_cluster'].keys())
    
#     path = []
#     for iteration in sorted(results['roi_to_cluster'].keys()):
#         if iteration > target_iteration:
#             break
#         if roi_index in results['roi_to_cluster'][iteration]:
#             path.append(results['roi_to_cluster'][iteration][roi_index])
    
#     return " -> ".join(path)
# %%



# Run the hierarchical clustering
results = hierarchical_clustering(data, max_iterations=3)

# Print results
print_hierarchy(results)
# %%

def get_roi_full_path(results, roi_index, target_iteration=None):
    """Get the full cluster path for a specific ROI"""
    if target_iteration is None:
        target_iteration = max(results['roi_to_cluster'].keys())
    
    path = []
    for iteration in sorted(results['roi_to_cluster'].keys()):
        if iteration > target_iteration:
            break
        if roi_index in results['roi_to_cluster'][iteration]:
            path.append(results['roi_to_cluster'][iteration][roi_index])
    
    return " -> ".join(path)
# Example: Get cluster path for specific ROIs
print("\n=== EXAMPLE ROI PATHS ===")
for roi_idx in [0, 10, 50, 100][:min(4, data['dFF_clean'].shape[0])]:
    if roi_idx in results['filtered_indices']:
        path = get_roi_full_path(results, roi_idx)
        print(f"ROI {roi_idx}: {path}")

# %%

def get_roi_full_path(results, roi_index, target_iteration=None):
    """Get the full cluster path for a specific ROI"""
    if target_iteration is None:
        target_iteration = max(results['roi_to_cluster'].keys())
    
    path = []
    for iteration in sorted(results['roi_to_cluster'].keys()):
        if iteration > target_iteration:
            break
        if roi_index in results['roi_to_cluster'][iteration]:
            path.append(results['roi_to_cluster'][iteration][roi_index])
    return " -> ".join(path)
    # if path[-1] != 'noise':
    #     return path[-1]
    # elif len(path) > 1: 
    #     path = path[-2] + '.' + path[-1]
    #     return path
    # else:
    #     return path[-1]



# %%
print("\n=== EXAMPLE ROI PATHS ===")
paths = []
rois = [0, 1, 6, 8, 10, 50, 100, 200, 18 ,477, 633]
rois = results['filtered_indices']
for roi_idx in range(len(rois)):
    if roi_idx in results['filtered_indices']:
        path = get_roi_full_path(results, roi_idx)
        paths.append(path)

for roi_idx in range(len(paths)):
    print(f"ROI {rois[roi_idx]}: {paths[roi_idx]}")


unique_clusters = list(dict.fromkeys(paths))
print(unique_clusters)

print(unique_clusters[3].split(" -> "))

cluster_indices = []
for idx in range(len(unique_clusters)):
    cluster_indices.append(unique_clusters[idx].split(" -> "))

print(cluster_indices)

# for cluster_idx in cluster_indices:
#     for idx in cluster_idx:
        
unique_cluster_rois = []
for cluster in unique_clusters:
    # indices where element == 'a'
    indices = [i for i, x in enumerate(paths) if x == cluster]
    # print(indices)   # [0, 3, 5]
    unique_cluster_rois.append(indices)



# %%


def plot_cluster_traces(data, clusters, cluster_labels, start_time=0, n_samples=5, duration=10):
    """
    Plot sample traces from each cluster in separate figures, with each trace as its own subplot
    
    Parameters:
    -----------
    data : dict
        Contains 'dFF_clean' (n_rois x n_timepoints) and 'imaging_time' (timepoints,)
    clusters : array
        Length n_rois, indicates which cluster each ROI belongs to
    cluster_labels : dict
        Maps cluster IDs to their string labels
    start_time : float
        Start time in seconds for the segment to plot
    n_samples : int
        Number of sample traces to plot per cluster
    duration : float
        Duration in seconds of segment to plot
    """
    
    # Get time segment
    time = data['imaging_time']
    end_time = start_time + duration
    time_mask = (time >= start_time) & (time <= end_time)
    time_segment = time[time_mask]
    
    # Get unique cluster IDs
    # unique_clusters = np.unique(clusters)
    
    # Create separate figure for each cluster
    for idx, cluster_rois in enumerate(clusters):
        # Get ROIs belonging to this cluster
        # cluster_mask = clusters == cluster_id
        # cluster_rois = np.where(cluster_mask)[0]
        cluster_id = cluster_labels[idx]
        # Select random sample of ROIs from this cluster
        n_rois_in_cluster = len(cluster_rois)
        sample_size = min(n_samples, n_rois_in_cluster)
        sampled_rois = np.random.choice(cluster_rois, size=sample_size, replace=False)
        
        # Create figure with subplots for each trace
        fig, axes = plt.subplots(sample_size, 1, figsize=(12, 2*sample_size), sharex=True)
        if sample_size == 1:
            axes = [axes]
        
        # Get cluster label
        cluster_label = cluster_id
        # try:
        #     cluster_label = cluster_labels[cluster_id] if cluster_id < len(cluster_labels) else f'Cluster {cluster_id}'
        # except (IndexError, TypeError):
        #     cluster_label = f'Cluster {cluster_id}'
        
        # Plot each trace in its own subplot
        for i, roi_idx in enumerate(sampled_rois):
            trace_segment = data['dFF_clean'][roi_idx, time_mask]
            axes[i].plot(time_segment, trace_segment, linewidth=1.5, color='blue')
            axes[i].set_ylabel('dF/F')
            axes[i].set_title(f'ROI {roi_idx}')
            axes[i].grid(True, alpha=0.3)
        
        # Set overall figure title and x-label
        fig.suptitle(f'{cluster_label} (n={n_rois_in_cluster}, showing {sample_size} traces)', 
                     fontsize=14, y=0.98)
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for suptitle
        plt.show()



plot_cluster_traces(data, unique_cluster_rois, unique_clusters, start_time=50, n_samples=20, duration=10)




# %%


















# %%

def plot_subcluster_traces(data, results, start_time=0, n_samples=5, duration=10):
    """
    Plot sample traces from each subcluster in separate figures, with each trace as its own subplot
    
    Parameters:
    -----------
    data : dict
        Contains 'dFF_clean' (n_rois x n_timepoints) and 'imaging_time' (timepoints,)
    results : dict
        Hierarchical clustering results with structure:
        results[parent_cluster_id]['subclusters'][subcluster_id] = {'rois': [...], 'label': '...'}
    start_time : float
        Start time in seconds for the segment to plot
    n_samples : int
        Number of sample traces to plot per subcluster
    duration : float
        Duration in seconds of segment to plot
    """
    
    # Get time segment
    time = data['imaging_time']
    end_time = start_time + duration
    time_mask = (time >= start_time) & (time <= end_time)
    time_segment = time[time_mask]
    
    # Iterate through all parent clusters and their subclusters
    for parent_id, parent_data in results.items():
        if 'subclusters' not in parent_data:
            continue
            
        subclusters = parent_data['subclusters']
        
        for subcluster_id, subcluster_data in subclusters.items():
            # Get ROIs belonging to this subcluster
            cluster_rois = np.array(subcluster_data['rois'])
            
            if len(cluster_rois) == 0:
                continue
                
            # Select random sample of ROIs from this subcluster
            n_rois_in_cluster = len(cluster_rois)
            sample_size = min(n_samples, n_rois_in_cluster)
            sampled_rois = np.random.choice(cluster_rois, size=sample_size, replace=False)
            
            # Create figure with subplots for each trace
            fig, axes = plt.subplots(sample_size, 1, figsize=(12, 2*sample_size), sharex=True)
            if sample_size == 1:
                axes = [axes]
            
            # Get subcluster label
            subcluster_label = subcluster_data.get('label', f'Parent {parent_id} - Subcluster {subcluster_id}')
            
            # Plot each trace in its own subplot
            for i, roi_idx in enumerate(sampled_rois):
                trace_segment = data['dFF_clean'][roi_idx, time_mask]
                axes[i].plot(time_segment, trace_segment, linewidth=1.5, color='blue')
                axes[i].set_ylabel('dF/F')
                axes[i].set_title(f'ROI {roi_idx}')
                axes[i].grid(True, alpha=0.3)
            
            # Set overall figure title and x-label
            fig.suptitle(f'{subcluster_label} (n={n_rois_in_cluster}, showing {sample_size} traces)', 
                         fontsize=14, y=0.98)
            axes[-1].set_xlabel('Time (s)')
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)  # Make room for suptitle
            plt.show()




plot_subcluster_traces(data, results, start_time=0, n_samples=5, duration=10)

# %%


# %%
# Get summary information
summary = get_subcluster_summary(results)
print("\nSubcluster Summary:")
for iteration, clusters in summary.items():
    print(f"Iteration {iteration}:")
    for cluster_key, info in clusters.items():
        print(f"  {cluster_key}: {info['size']} ROIs (parent: {info['parent']})")


# %%


def explore_clustering_results(results):
    """
    Traverse and explore the hierarchical clustering results structure
    to understand how to plot subcluster traces.
    """
    print("=== Clustering Results Structure ===\n")
    
    # Show overall structure
    print("Top-level keys:", list(results.keys()))
    print(f"Filtered indices: {len(results['filtered_indices'])} ROIs")
    print()
    
    # Traverse each iteration
    for iteration in sorted(results['hierarchy'].keys()):
        print(f"--- ITERATION {iteration} ---")
        
        iteration_clusters = results['hierarchy'][iteration]
        iteration_info = results['cluster_info'][iteration]
        
        print(f"Number of clusters: {len(iteration_clusters)}")
        
        for cluster_key, roi_indices in iteration_clusters.items():
            info = iteration_info[cluster_key]
            
            print(f"  Cluster '{cluster_key}':")
            print(f"    - ROI indices: {roi_indices[:5]}{'...' if len(roi_indices) > 5 else ''}")
            print(f"    - Size: {info['size']}")
            print(f"    - Parent: {info['parent']}")
            print(f"    - Confidence kept: {info['confidence_kept']}")
            print(f"    - Mean probability: {np.mean(info['probabilities']):.3f}")
            
            # Show hierarchical relationships
            if '.' in cluster_key:  # This is a subcluster
                parent_key = '.'.join(cluster_key.split('.')[:-1])
                print(f"    - This is a subcluster of: {parent_key}")
        print()

def plot_subcluster_traces(data, results, iteration_to_plot=0, max_clusters=6):
    """
    Plot traces for clusters from a specific iteration.
    This shows how to use the results dict for plotting.
    """
    X = data['dFF_clean']
    
    # Get clusters from specified iteration
    if iteration_to_plot not in results['hierarchy']:
        print(f"Iteration {iteration_to_plot} not found!")
        return
    
    clusters = results['hierarchy'][iteration_to_plot]
    cluster_info = results['cluster_info'][iteration_to_plot]
    
    # Filter out noise and sort by size
    valid_clusters = [(k, v) for k, v in clusters.items() if not k.endswith('noise')]
    valid_clusters.sort(key=lambda x: len(x[1]), reverse=True)
    
    # Plot top clusters
    n_clusters = min(len(valid_clusters), max_clusters)
    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 2*n_clusters))
    if n_clusters == 1:
        axes = [axes]
    
    for i, (cluster_key, roi_indices) in enumerate(valid_clusters[:n_clusters]):
        ax = axes[i]
        
        # Get cluster traces
        cluster_traces = X[roi_indices]
        
        # Plot individual traces (faded)
        for trace in cluster_traces:
            ax.plot(trace, alpha=0.3, color='gray', linewidth=0.5)
        
        # Plot mean trace (bold)
        mean_trace = np.mean(cluster_traces, axis=0)
        ax.plot(mean_trace, color='red', linewidth=2, label='Mean')
        
        # Add cluster info to title
        info = cluster_info[cluster_key]
        title = f"Cluster {cluster_key} (n={info['size']}, parent={info['parent']})"
        ax.set_title(title)
        ax.set_ylabel('dF/F')
        
        if i == n_clusters - 1:
            ax.set_xlabel('Time')
    
    plt.tight_layout()
    plt.suptitle(f'Iteration {iteration_to_plot} Clusters', y=1.02)
    plt.show()

# Usage example:
# results = hierarchical_clustering(data)
explore_clustering_results(results)
# plot_subcluster_traces(data, results, iteration_to_plot=0)
# plot_subcluster_traces(data, results, iteration_to_plot=1)  # subclusters






# %%

# import matplotlib.pyplot as plt
# import numpy as np

# def plot_cluster_traces(data, clusters, cluster_labels, start_time=0, n_samples=5, duration=10):
#     """
#     Plot sample traces from each cluster in separate subplots
    
#     Parameters:
#     -----------
#     data : dict
#         Contains 'dFF_clean' (n_rois x n_timepoints) and 'imaging_time' (timepoints,)
#     clusters : array
#         Length n_rois, indicates which cluster each ROI belongs to
#     cluster_labels : dict
#         Maps cluster IDs to their string labels
#     start_time : float
#         Start time in seconds for the segment to plot
#     n_samples : int
#         Number of sample traces to plot per cluster
#     duration : float
#         Duration in seconds of segment to plot
#     """
    
#     # Get time segment
#     time = data['imaging_time']
#     end_time = start_time + duration
#     time_mask = (time >= start_time) & (time <= end_time)
#     time_segment = time[time_mask]
    
#     # Get unique cluster IDs
#     unique_clusters = np.unique(clusters)
#     n_clusters = len(unique_clusters)
    
#     # Create subplots
#     fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 3*n_clusters), sharex=True)
#     if n_clusters == 1:
#         axes = [axes]
    
#     for i, cluster_id in enumerate(unique_clusters):
#         # Get ROIs belonging to this cluster
#         cluster_mask = clusters == cluster_id
#         cluster_rois = np.where(cluster_mask)[0]
        
#         # Select random sample of ROIs from this cluster
#         n_rois_in_cluster = len(cluster_rois)
#         sample_size = min(n_samples, n_rois_in_cluster)
#         sampled_rois = np.random.choice(cluster_rois, size=sample_size, replace=False)
        
#         # Plot traces for sampled ROIs
#         for roi_idx in sampled_rois:
#             trace_segment = data['dFF_clean'][roi_idx, time_mask]
#             axes[i].plot(time_segment, trace_segment, alpha=0.7, linewidth=1)
        
#         # Format subplot

#         # With this:
#         try:
#             cluster_label = cluster_labels[cluster_id] if cluster_id < len(cluster_labels) else f'Cluster {cluster_id}'
#         except (IndexError, TypeError):
#             cluster_label = f'Cluster {cluster_id}'
#         axes[i].set_ylabel('dF/F')
#         axes[i].set_title(f'{cluster_label} (n={n_rois_in_cluster}, showing {sample_size} traces)')
#         axes[i].grid(True, alpha=0.3)
    
#     axes[-1].set_xlabel('Time (s)')
#     plt.tight_layout()
#     plt.show()

# # Example usage:
# plot_cluster_traces(data, clusters, cluster_labels, start_time=50, n_samples=5, duration=10)





# %%

import matplotlib.pyplot as plt
import numpy as np

def plot_cluster_trace_segments(data, cluster_labels, n_clusters=None, segment_duration=10, 
                               traces_per_cluster=3, figsize=(15, 10)):
    """
    Plot 10-second segments from traces in each cluster to examine cluster characteristics
    
    Parameters:
    - data: dictionary containing 'dFF_clean' and 'imaging_time'
    - cluster_labels: array of cluster assignments for each trace
    - n_clusters: number of clusters to plot (if None, plot all)
    - segment_duration: duration of segments to plot in seconds
    - traces_per_cluster: number of traces to show per cluster
    """
    
    traces = data['dFF_clean']
    time_vec = data['imaging_time']
    
    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    if n_clusters is not None:
        unique_clusters = unique_clusters[:n_clusters]
    
    # Calculate sampling rate
    dt = np.mean(np.diff(time_vec))
    segment_samples = int(segment_duration / dt)
    
    # Create subplots
    n_rows = len(unique_clusters)
    fig, axes = plt.subplots(n_rows, traces_per_cluster, figsize=figsize, 
                            sharex=True, sharey=True)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for cluster_idx, cluster_id in enumerate(unique_clusters):
        # Get traces belonging to this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_trace_indices = np.where(cluster_mask)[0]
        
        # Randomly select traces from this cluster
        n_available = len(cluster_trace_indices)
        n_to_plot = min(traces_per_cluster, n_available)
        selected_indices = np.random.choice(cluster_trace_indices, n_to_plot, replace=False)
        
        for trace_idx in range(traces_per_cluster):
            ax = axes[cluster_idx, trace_idx]
            
            if trace_idx < n_to_plot:
                trace_id = selected_indices[trace_idx]
                trace = traces[trace_id, :]
                
                # Find a good segment (avoid NaNs if possible)
                valid_starts = []
                for start_idx in range(0, len(trace) - segment_samples, segment_samples//2):
                    segment = trace[start_idx:start_idx + segment_samples]
                    if not np.any(np.isnan(segment)):
                        valid_starts.append(start_idx)
                
                if valid_starts:
                    start_idx = np.random.choice(valid_starts)
                else:
                    # If no valid segment found, just use a random start
                    start_idx = np.random.randint(0, max(1, len(trace) - segment_samples))
                
                end_idx = start_idx + segment_samples
                
                # Extract segment
                time_segment = time_vec[start_idx:end_idx] - time_vec[start_idx]
                trace_segment = trace[start_idx:end_idx]
                
                # Plot
                ax.plot(time_segment, trace_segment, 'b-', linewidth=1)
                ax.set_title(f'Cluster {cluster_id}\nTrace {trace_id}', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                if trace_idx == 0:
                    ax.set_ylabel('dF/F', fontsize=10)
                if cluster_idx == len(unique_clusters) - 1:
                    ax.set_xlabel('Time (s)', fontsize=10)
            else:
                ax.set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'10-second segments from traces in each cluster', fontsize=14, y=1.02)
    return fig, axes

def plot_cluster_average_traces(data, cluster_labels, segment_duration=30, figsize=(15, 8)):
    """
    Plot average traces for each cluster over longer segments
    """
    traces = data['dFF_clean']
    time_vec = data['imaging_time']
    
    unique_clusters = np.unique(cluster_labels)
    dt = np.mean(np.diff(time_vec))
    segment_samples = int(segment_duration / dt)
    
    # Find a good common time segment
    start_idx = len(time_vec) // 4  # Start at 1/4 through the recording
    end_idx = start_idx + segment_samples
    
    if end_idx > len(time_vec):
        end_idx = len(time_vec)
        start_idx = end_idx - segment_samples
    
    time_segment = time_vec[start_idx:end_idx] - time_vec[start_idx]
    
    fig, axes = plt.subplots(2, len(unique_clusters)//2 + len(unique_clusters)%2, 
                            figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()
    
    for cluster_idx, cluster_id in enumerate(unique_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_traces = traces[cluster_mask, start_idx:end_idx]
        
        ax = axes[cluster_idx]
        
        # Plot individual traces in light gray
        for trace in cluster_traces:
            if not np.all(np.isnan(trace)):
                ax.plot(time_segment, trace, color='gray', alpha=0.3, linewidth=0.5)
        
        # Plot average trace
        avg_trace = np.nanmean(cluster_traces, axis=0)
        ax.plot(time_segment, avg_trace, 'r-', linewidth=2, label='Average')
        
        ax.set_title(f'Cluster {cluster_id} (n={np.sum(cluster_mask)})', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if cluster_idx >= len(unique_clusters) - 2:
            ax.set_xlabel('Time (s)')
        if cluster_idx % 2 == 0:
            ax.set_ylabel('dF/F')
    
    # Hide unused subplots
    for idx in range(len(unique_clusters), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'{segment_duration}s segments: Individual traces (gray) and cluster averages (red)', 
                fontsize=14, y=1.02)
    return fig, axes



# Example workflow:
print("Data shape:", data['dFF_clean'].shape)
print("Time vector shape:", data['imaging_time'].shape)
print("Time range:", data['imaging_time'][0], "to", data['imaging_time'][-1], "seconds")

# Now plot the segments
fig1, axes1 = plot_cluster_trace_segments(data, cluster_labels, segment_duration=10, traces_per_cluster=4)
plt.show()

fig2, axes2 = plot_cluster_average_traces(data, cluster_labels, segment_duration=30)
plt.show()





# %%
# DIAGNOSE BASELINE AND dF/F ISSUES
print("\n=== DIAGNOSING BASELINE AND dF/F ISSUES ===")

# Check intermediate values
print(f"Intermediate processing statistics:")
print(f"  Fc: min={np.min(data['Fc']):.1f}, max={np.max(data['Fc']):.1f}, std={np.std(data['Fc']):.1f}")
print(f"  F0: min={np.min(data['F0']):.1f}, max={np.max(data['F0']):.1f}, std={np.std(data['F0']):.1f}")

# Check for near-zero F0 values (causes dF/F explosions)
near_zero_f0 = np.sum(data['F0'] < 0.1)
negative_f0 = np.sum(data['F0'] < 0)

print(f"Baseline (F0) quality check:")
print(f"  F0 values < 0.1: {near_zero_f0} out of {data['F0'].size} ({100*near_zero_f0/data['F0'].size:.1f}%)")
print(f"  Negative F0 values: {negative_f0} out of {data['F0'].size} ({100*negative_f0/data['F0'].size:.1f}%)")

if near_zero_f0 > 0 or negative_f0 > 0:
    print(f"  Min F0 value: {np.min(data['F0']):.6f}")
    
    # Find which ROIs have problematic F0
    min_f0_per_roi = np.min(data['F0'], axis=1)
    problem_f0_rois = np.where(min_f0_per_roi < 0.1)[0]
    print(f"  ROIs with problematic F0: {problem_f0_rois[:10]}{'...' if len(problem_f0_rois) > 10 else ''}")

# Check for exploded dF/F values
dff_clean = data['dFF_clean']
n_rois = dff_clean.shape[0]

exploded_rois = []
extreme_rois = []

print(f"\ndF/F quality check:")
for roi_idx in range(n_rois):
    roi_trace = dff_clean[roi_idx, :]
    
    roi_min = np.min(roi_trace)
    roi_max = np.max(roi_trace)
    roi_std = np.std(roi_trace)
    roi_range = roi_max - roi_min
    
    # Flag ROIs with extreme values
    if roi_max > 50 or roi_min < -50:
        exploded_rois.append({
            'roi': roi_idx,
            'min': roi_min,
            'max': roi_max,
            'std': roi_std,
            'range': roi_range
        })
    elif roi_max > 7 or roi_min < -3 or roi_std > 5:
        extreme_rois.append({
            'roi': roi_idx,
            'min': roi_min,
            'max': roi_max,
            'std': roi_std,
            'range': roi_range
        })

print(f"  Total ROIs: {n_rois}")
print(f"  Exploded ROIs (>50 or <-50): {len(exploded_rois)}")
print(f"  Extreme ROIs (>7, <-3, std>5): {len(extreme_rois)}")

if len(exploded_rois) > 0:
    print(f"\n  Worst exploded ROIs:")
    exploded_sorted = sorted(exploded_rois, key=lambda x: x['range'], reverse=True)
    for roi_info in exploded_sorted[:5]:  # Show top 5
        print(f"    ROI {roi_info['roi']:3d}: range {roi_info['range']:8.1f} "
              f"({roi_info['min']:6.1f} to {roi_info['max']:6.1f}), std {roi_info['std']:6.1f}")

if len(extreme_rois) > 0:
    print(f"\n  Worst extreme ROIs:")
    extreme_sorted = sorted(extreme_rois, key=lambda x: x['range'], reverse=True)
    for roi_info in extreme_sorted[:5]:  # Show top 5
        print(f"    ROI {roi_info['roi']:3d}: range {roi_info['range']:8.1f} "
              f"({roi_info['min']:6.1f} to {roi_info['max']:6.1f}), std {roi_info['std']:6.1f}")

# Overall dF/F statistics
print(f"\n  Overall dF/F statistics:")
print(f"    Range: {np.min(dff_clean):.3f} to {np.max(dff_clean):.3f}")
print(f"    Mean: {np.mean(dff_clean):.3f}")
print(f"    Std: {np.std(dff_clean):.3f}")

# Flag if we have major issues
total_problem_rois = len(exploded_rois) + len(extreme_rois)
if total_problem_rois > n_rois * 0.1:  # More than 10% problematic
    print(f"\n  *** WARNING: {total_problem_rois}/{n_rois} ({100*total_problem_rois/n_rois:.1f}%) ROIs have problematic dF/F values ***")
    print(f"  This suggests baseline calculation issues - consider:")
    print(f"    1. Using session-wide baseline instead of rolling baseline")
    print(f"    2. Adding baseline protection (minimum threshold)")
    print(f"    3. Checking for movement artifacts or photobleaching")
elif total_problem_rois > 0:
    print(f"\n  Note: {total_problem_rois}/{n_rois} ({100*total_problem_rois/n_rois:.1f}%) ROIs have elevated dF/F values")
    print(f"  This is within acceptable range but monitor for analysis impact")
else:
    print(f"\n  ✓ All ROIs have reasonable dF/F values")





# %%








# Add this to your diagnostic section
if len(exploded_rois) > 0:
    worst_roi_indices = [r['roi'] for r in exploded_sorted[:10]]
    diagnose_roi_explosion_types(data, worst_roi_indices)








# %%

# In your baseline calculation, add:
F0_safe = np.maximum(F0, 0.1 * np.mean(F, axis=1, keepdims=True))

# Replace rolling baseline with session-wide:
F0_session = np.percentile(F, 10, axis=1, keepdims=True)
F0_session = F0_session + 0.01 * np.mean(F0_session, axis=1, keepdims=True)

# Use rolling baseline but with session-wide floor:
F0_rolling = your_current_calculation()
F0_floor = np.percentile(F, 5, axis=1, keepdims=True) * 0.5  # Conservative floor
F0_protected = np.maximum(F0_rolling, F0_floor)

# %%
# Add this right after line 8747 (after compute_baseline_and_dff)
exploded_rois, extreme_rois = find_exploded_rois(data)

# Also check the intermediate values
print(f"\n=== INTERMEDIATE VALUES CHECK ===")
print(f"Fc stats: min={np.min(data['Fc']):.1f}, max={np.max(data['Fc']):.1f}, std={np.std(data['Fc']):.1f}")
print(f"F0 stats: min={np.min(data['F0']):.1f}, max={np.max(data['F0']):.1f}, std={np.std(data['F0']):.1f}")

# Check for near-zero F0 values
near_zero_f0 = np.sum(data['F0'] < 0.1)
print(f"F0 values < 0.1: {near_zero_f0} out of {data['F0'].size} ({100*near_zero_f0/data['F0'].size:.1f}%)")

if near_zero_f0 > 0:
    print(f"Min F0 value: {np.min(data['F0']):.6f}")
    
    # Find which ROIs have near-zero F0
    min_f0_per_roi = np.min(data['F0'], axis=1)
    problem_rois = np.where(min_f0_per_roi < 0.1)[0]
    print(f"ROIs with F0 < 0.1: {problem_rois[:10]}... (showing first 10)")










# %%
compute_qc_metrics(data, cfg)


# %%
# STEP 8: Summary
summarize(data, cfg)
quick_plot(data, cfg, roi=0)

# %%
# STEP 6: Detailed review plots
prepare_review_cache(data, cfg)
_ = plot_roi_review(data, cfg, roi=6, save=False)

# %%
# STEP 7: Batch export (optional)
batch_plot_rois(data, cfg, list(range(data['dFF'].shape[0])), limit=30)
# batch_plot_rois(data, cfg, [2,18], limit=30)
# batch_plot_rois(data, cfg, list(range(445,485)), limit=None)


print("\n" + "="*50)
print("PIPELINE COMPLETE")
print("="*50)






# %%
cfg = load_cfg_yaml(cfg_path)
sid_img_data = load_sid_imaging_preprocess(cfg)
data.update(sid_img_data)

# %%
session_data = load_sid_behavioral_preprocess(cfg)
data.update(session_data)


# %%


# After loading both imaging and behavioral data, align the timestamps
data = align_trial_timestamps_to_vol_start(data, cfg, tolerance_s=0.05)

# Plot diagnostic information
plot_trial_alignment_diagnostic(data, cfg, show_n_trials=400, save=True)
    
    


# %%
# FINAL STEP: Save complete processed data 
save_path = save_sid_imaging_data(data, cfg)
print(f"\n{'='*50}")
print(f"SID IMAGING DATA SAVED: {save_path}")
print(f"{'='*50}")

# Test the helper functions
print(f"\nTesting ROI index helpers:")
print(f"Filtered ROI 0 -> Original Suite2p ROI {get_original_suite2p_roi_index(data, 0)}")
print(f"Filtered ROI 5 -> Original Suite2p ROI {get_original_suite2p_roi_index(data, 5)}")

# Test reverse lookup
orig_idx = get_original_suite2p_roi_index(data, 0)
filtered_idx = get_filtered_roi_index(data, orig_idx)
print(f"Round trip test: filtered 0 -> original {orig_idx} -> filtered {filtered_idx}")








# %%
# Test batch plotting with a small subset
test_trials = list(range(20, 30))  # Just first 3 trials
test_rois = [2]       # Just 2 ROIs

print(f"Testing batch plotting: {len(test_trials)} trials × {len(test_rois)} ROIs")

try:
    batch_plot_single_trials(
        data, cfg,
        trial_indices=test_trials,
        rois=test_rois,
        pre_trial_sec=3.0,
        post_trial_sec=7.0
    )
    print("Batch single trial plotting successful!")
    
except Exception as e:
    print(f"Error in batch single trial plotting: {e}")
    import traceback
    traceback.print_exc()
  

  
# %%



# Test batch plotting with a small subset
test_trials = list(range(20, 30))  # Just first 3 trials
test_rois = [5]       # Just 2 ROIs

print(f"Testing batch plotting: {len(test_trials)} trials × {len(test_rois)} ROIs")

try:
    batch_plot_single_trials(
        data, cfg,
        trial_indices=test_trials,
        rois=test_rois,
        pre_trial_sec=3.0,
        post_trial_sec=7.0
    )
    print("Batch single trial plotting successful!")
    
except Exception as e:
    print(f"Error in batch single trial plotting: {e}")
    import traceback
    traceback.print_exc()    
    
    
    
# %%



# Test batch plotting with a small subset
test_trials = list(range(20, 30))  # Just first 3 trials
test_rois = [18]       # Just 2 ROIs

print(f"Testing batch plotting: {len(test_trials)} trials × {len(test_rois)} ROIs")

try:
    batch_plot_single_trials(
        data, cfg,
        trial_indices=test_trials,
        rois=test_rois,
        pre_trial_sec=3.0,
        post_trial_sec=7.0
    )
    print("Batch single trial plotting successful!")
    
except Exception as e:
    print(f"Error in batch single trial plotting: {e}")
    import traceback
    traceback.print_exc()    
    
    
    
    
# %%



# Test batch plotting with a small subset
test_trials = list(range(100, 110))  # Just first 3 trials
test_rois = [477]       # Just 2 ROIs

print(f"Testing batch plotting: {len(test_trials)} trials × {len(test_rois)} ROIs")

try:
    batch_plot_single_trials(
        data, cfg,
        trial_indices=test_trials,
        rois=test_rois,
        pre_trial_sec=3.0,
        post_trial_sec=7.0
    )
    print("Batch single trial plotting successful!")
    
except Exception as e:
    print(f"Error in batch single trial plotting: {e}")
    import traceback
    traceback.print_exc()  
    
    
    
    



    
    
