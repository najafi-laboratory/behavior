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
    """STEP 1: Load raw Suite2p outputs"""
    print("\n=== LOADING SUITE2P DATA ===")
    folder = cfg["paths"]["plane_dir"]
    memmap = cfg["io"]["memmap"]
    copy_on_load = cfg["io"]["copy_on_load"]
    
    print(f"Source folder: {folder}")
    print(f"Memmap mode: {memmap}, Copy on load: {copy_on_load}")
    
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Suite2p folder not found: {folder}")
    
    data: Dict[str, Any] = {}
    
    # Load core arrays
    data["ops"]    = _load_npy_safe(os.path.join(folder, "ops.npy"), memmap=False)
    data["F"]      = _load_npy_safe(os.path.join(folder, "F.npy"), memmap)
    data["Fneu"]   = _load_npy_safe(os.path.join(folder, "Fneu.npy"), memmap)
    data["spks"]   = _load_npy_safe(os.path.join(folder, "spks.npy"), memmap)
    data["iscell"] = _load_npy_safe(os.path.join(folder, "iscell.npy"), memmap=False)
    data["stat"]   = _load_npy_safe(os.path.join(folder, "stat.npy"), memmap=False)
    
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

# def compute_baseline_and_dff(data: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
#     """STEP 4B: Baseline estimation and dF/F calculation"""
#     print("\n=== BASELINE & dF/F COMPUTATION ===")
    
#     Fc = data["Fc"]
#     fs = float(cfg["acq"]["fs"])
#     base_cfg = cfg["baseline"]
    
#     win_s = base_cfg["win_s"]
#     prc = base_cfg["percentile"]
#     sig_s = base_cfg["smooth_sigma_s"]
#     div = base_cfg["smooth_sigma_divisor"]
#     eps = base_cfg["f0_epsilon"]
    
#     print(f"Baseline parameters:")
#     print(f"  Window: {win_s}s ({int(win_s * fs)} samples)")
#     print(f"  Percentile: {prc}")
#     print(f"  Smoothing sigma: {sig_s}s")
#     print(f"  F0 epsilon: {eps}")
    
#     win = max(1, int(win_s * fs))
#     N, T = Fc.shape
    
#     # Create windows
#     edges = np.arange(0, T, win)
#     if edges[-1] != T:
#         edges = np.append(edges, T)
    
#     n_windows = len(edges) - 1
#     print(f"  Created {n_windows} windows for baseline estimation")
    
#     F0 = np.empty_like(Fc, dtype=np.float32)
    
#     print("Computing baselines...")
#     for i in range(N):
#         if i % 100 == 0 and i > 0:
#             print(f"    Processed {i}/{N} ROIs...")
        
#         centers, vals = [], []
#         for j in range(len(edges) - 1):
#             s, e = edges[j], edges[j + 1]
#             seg = Fc[i, s:e]
#             if seg.size == 0:
#                 continue
#             centers.append(0.5 * (s + e - 1))
#             vals.append(np.nanpercentile(seg, prc))
        
#         if len(vals) == 0:
#             F0[i] = 0
#         elif len(vals) == 1:
#             F0[i] = vals[0]
#         else:
#             F0[i] = np.interp(np.arange(T), centers, vals).astype(np.float32)
        
#         # Smoothing
#         if sig_s > 0:
#             sigma = (sig_s * fs) / div
#             F0[i] = gaussian_filter(F0[i], sigma=sigma)
    
#     # Compute dF/F
#     dF = Fc - F0
#     with np.errstate(divide='ignore', invalid='ignore'):
#         dFF = dF / (F0 + eps)
    
#     data["F0"] = F0
#     data["dF"] = dF
#     data["dFF"] = dFF
    
#     # Outlier cleaning
#     o_cfg = cfg["outliers"]
#     if o_cfg["enable"]:
#         print("Applying outlier cleaning...")
#         thr = o_cfg["z_thr"]
#         print(f"  Z-score threshold: {thr}")
        
#         dFF_clean = np.empty_like(dFF)
#         total_outliers = 0
        
#         for i in range(dFF.shape[0]):
#             cleaned = fill_outliers_neighbor_mean(dFF[i], thr=thr)
#             outliers = np.sum(cleaned != dFF[i])
#             total_outliers += outliers
#             dFF_clean[i] = cleaned
        
#         data["dFF_clean"] = dFF_clean
#         print(f"  Outliers cleaned: {total_outliers} timepoints across all ROIs")
#     else:
#         print("Outlier cleaning disabled")
#         data["dFF_clean"] = dFF.copy()
    
#     # Summary statistics
#     dff_final = data["dFF_clean"]
#     max_dff = float(np.nanmax(np.abs(dff_final)))
#     mean_f0 = float(np.nanmean(F0))
    
#     print(f"Baseline & dF/F computation complete:")
#     print(f"  Mean F0: {mean_f0:.1f}")
#     print(f"  Max |dF/F|: {max_dff:.3f}")
#     print(f"  dF/F shape: {dff_final.shape}")
    
#     return data

def compute_percentile_baseline(data: Dict[str, Any], cfg: Dict[str, Any]) -> np.ndarray:
    """Compute F0 baseline using percentile windowing (your current sophisticated method)"""
    print("  Computing percentile-based baseline...")
    
    Fc = data["Fc"]
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
    N, T = Fc.shape
    
    # Create windows
    edges = np.arange(0, T, win)
    if edges[-1] != T:
        edges = np.append(edges, T)
    
    n_windows = len(edges) - 1
    print(f"    Created {n_windows} windows for baseline estimation")
    
    F0 = np.empty_like(Fc, dtype=np.float32)
    
    for i in range(N):
        if i % 100 == 0 and i > 0:
            print(f"      Processed {i}/{N} ROIs...")
        
        centers, vals = [], []
        for j in range(len(edges) - 1):
            s, e = edges[j], edges[j + 1]
            seg = Fc[i, s:e]
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

def compute_baseline_and_dff(data: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """STEP 4B: Baseline estimation, dF/F calculation, and spike deconvolution"""
    Fc = data["Fc"]
    if Fc is None:
        raise ValueError("Fc required for baseline computation - run neuropil regression first")
    
    fs = float(cfg["acq"]["fs"])
    base_cfg = cfg["baseline"]
    method = base_cfg["method"]
    
    print(f"Baseline method: {method}")
    
    # Compute baseline using selected method
    if method == "custom":
        F0 = compute_percentile_baseline(data, cfg)
    elif method == "suite2p":
        F0 = compute_suite2p_baseline(data, cfg)
    else:
        raise ValueError(f"Unknown baseline method: {method}. Use 'custom' or 'suite2p'")
    
    # Compute dF/F
    print("  Computing dF/F...")
    eps = base_cfg.get("f0_epsilon", 1.0e-6)
    
    dF = Fc - F0
    # protect from divide by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        dFF = dF / (F0 + eps)
    
    data["F0"] = F0
    data["dF"] = dF
    data["dFF"] = dFF
    
    # Outlier cleaning
    o_cfg = cfg["outliers"]
    if o_cfg["enable"]:
        print("  Applying outlier cleaning...")
        thr = o_cfg["z_thr"]
        print(f"    Z-score threshold: {thr}")
        
        dFF_clean = np.empty_like(dFF)
        total_outliers = 0
        
        for i in range(dFF.shape[0]):
            cleaned = fill_outliers_neighbor_mean(dFF[i], thr=thr)
            outliers = np.sum(cleaned != dFF[i])
            total_outliers += outliers
            dFF_clean[i] = cleaned
        
        data["dFF_clean"] = dFF_clean
        print(f"    Outliers cleaned: {total_outliers} timepoints across all ROIs")
    else:
        print("  Outlier cleaning disabled")
        data["dFF_clean"] = dFF.copy()
    
    # Spike deconvolution - run both versions for comparison
    deconv_cfg = cfg.get("deconvolution", {})
    if deconv_cfg.get("enable", True):
        print("  Running OASIS spike deconvolution (both versions)...")
        
        # Version 1: Suite2p batch (spikes only)
        print("    Version 1: Suite2p batch API...")
        deconv_oasis_from_Fc(data, cfg)
        
        # Version 2: Individual trace reconstruction (full components)
        print("    Version 2: Individual trace reconstruction...")
        deconv_oasis_from_Fc_recon(data, cfg)
        
        # Compare results if both succeeded
        if "spks_oasis" in data and "spks_oasis_recon" in data:
            print("  Comparing deconvolution methods...")
            spks_batch = data["spks_oasis"]
            spks_recon = data["spks_oasis_recon"]
            
            # Basic comparison metrics
            corr_coef = np.corrcoef(spks_batch.flatten(), spks_recon.flatten())[0,1]
            batch_total = np.sum(spks_batch > 0)
            recon_total = np.sum(spks_recon > 0)
            
            print(f"    Spike correlation: {corr_coef:.4f}")
            print(f"    Batch spikes: {batch_total}, Recon spikes: {recon_total}")
            print(f"    Spike count ratio: {recon_total/max(batch_total,1):.3f}")
            
            # Check if we got the extra components from reconstruction
            if "C_oasis_recon" in data:
                print(f"    Reconstruction provides: spikes, denoised calcium, baselines, decay coefficients")
            else:
                print(f"    Reconstruction failed to provide additional components")
    else:
        print("  Spike deconvolution disabled")
    
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
def fill_outliers_neighbor_mean(x: np.ndarray, thr: float) -> np.ndarray:
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



# def detect_ca_transients_debug_enhanced(x, fs, roi_id=None, debug=True, **params):
#     """
#     Enhanced debug version that returns threshold information and detailed rejection tracking
#     """
#     if debug and roi_id is not None:
#         print(f"\n=== DEBUGGING {roi_id} ===")
    
#     x = np.asarray(x, float)
#     T = x.size
    
#     if debug:
#         print(f"Input: {T} frames, fs={fs:.1f}Hz, duration={T/fs:.1f}s")
#         print(f"dF/F range: [{np.nanmin(x):.3f}, {np.nanmax(x):.3f}], std={np.nanstd(x):.3f}")
    
#     # Extract parameters with defaults
#     smooth_sigma_s = params.get('smooth_sigma_s', 0.05)
#     k_on = params.get('k_on', 1.5)
#     k_off = params.get('k_off', 0.5)
#     min_amp = params.get('min_amp', 0.05)
#     min_series_peak_dff = params.get('min_series_peak_dff', 0.10)
    
#     if debug:
#         print(f"Parameters: k_on={k_on}, k_off={k_off}, min_amp={min_amp}")

#     if T < 2:
#         if debug: print("REJECT: Too few frames")
#         return [], np.zeros(T, dtype=bool), {'th_on': 0, 'th_off': 0, 'baseline': 0, 'sigma': 0}
    
#     # Check for invalid traces
#     if np.all(np.isnan(x)) or np.all(x == x[0]):
#         if debug: print("REJECT: Invalid trace (all NaN or constant)")
#         return [], np.zeros(T, dtype=bool), {'th_on': 0, 'th_off': 0, 'baseline': 0, 'sigma': 0}

#     # Smoothing
#     if smooth_sigma_s > 0:
#         sigma_frames = max(1, int(round(smooth_sigma_s * fs)))
#         x_s = gaussian_filter1d(x, sigma=sigma_frames)
#         if debug: print(f"Smoothed with sigma={sigma_frames} frames")
#     else:
#         x_s = x.copy()

#     # Noise estimation
#     try:
#         valid_mask = np.isfinite(x_s)
#         if not np.any(valid_mask):
#             if debug: print("REJECT: No valid data points")
#             return [], np.zeros(T, dtype=bool), {'th_on': 0, 'th_off': 0, 'baseline': 0, 'sigma': 0}
        
#         x_valid = x_s[valid_mask]
        
#         if len(x_valid) < 10:
#             if debug: print("REJECT: Too few valid points")
#             return [], np.zeros(T, dtype=bool), {'th_on': 0, 'th_off': 0, 'baseline': 0, 'sigma': 0}
        
#         # Simple noise estimation for debugging
#         median_val = np.median(x_valid)
        
#         # Use MAD for noise estimate
#         mad = np.median(np.abs(x_valid - median_val))
#         if mad < 1e-12:
#             mad = np.std(x_valid) / 1.4826
#             if mad < 1e-12:
#                 if debug: print("REJECT: Zero variance trace")
#                 return [], np.zeros(T, dtype=bool), {'th_on': 0, 'th_off': 0, 'baseline': median_val, 'sigma': 0}
        
#         sigma = 1.4826 * mad
        
#         if debug:
#             print(f"Noise estimation: median={median_val:.3f}, sigma={sigma:.3f}, mad={mad:.3f}")
        
#     except Exception as e:
#         if debug: print(f"REJECT: Noise estimation error: {e}")
#         return [], np.zeros(T, dtype=bool), {'th_on': 0, 'th_off': 0, 'baseline': 0, 'sigma': 0}

#     # Thresholds
#     y = x_s
#     th_on = median_val + k_on * sigma
#     th_off = median_val + k_off * sigma
    
#     if debug:
#         print(f"Thresholds: th_on={th_on:.3f}, th_off={th_off:.3f}")
#         above_on = np.sum(y > th_on)
#         above_off = np.sum(y > th_off)
#         print(f"Points above thresholds: {above_on} above th_on, {above_off} above th_off")

#     # Enhanced tracking of detection stages
#     debug_info = {
#         'th_on': th_on,
#         'th_off': th_off, 
#         'baseline': median_val,
#         'sigma': sigma,
#         'smoothed_trace': y.copy(),
#         'potential_onsets': [],
#         'rejected_reasons': {'no_peak': 0, 'low_amp': 0, 'no_offset': 0},
#         'rejection_details': [],  # Store details for each rejection
#         'final_events': []
#     }
    
#     events = []
#     mask = np.zeros(T, dtype=bool)
#     i = 1
#     onset_count = 0
    
#     # Main detection loop with detailed rejection tracking
#     while i < T:
#         if not np.isfinite(y[i-1]) or not np.isfinite(y[i]):
#             i += 1
#             continue
            
#         # ONSET detection
#         if y[i-1] < th_on and y[i] >= th_on:
#             onset_count += 1
#             i_on = i
            
#             # Store potential onset
#             debug_info['potential_onsets'].append({
#                 'i_on': i_on,
#                 't_on': i_on/fs,
#                 'value': y[i_on]
#             })
            
#             # FIXED: Find baseline - look backwards from onset to find last time below th_off
#             j = i_on - 1  # Start just before onset
#             while j >= 0 and np.isfinite(y[j]) and y[j] > th_off:
#                 j -= 1
#             i_base = max(0, j)  # Don't go negative
#             baseline = y[i_base]
            
#             # Quick amplitude check
#             search_end = min(T, i_on + int(2.0 * fs))  # 2 second search
#             if search_end > i_on:
#                 peak_val = np.nanmax(y[i_on:search_end])
#                 amp = peak_val - baseline
                
#                 if amp >= min_amp and amp >= min_series_peak_dff:
#                     # For debugging, create a simple event
#                     i_peak = i_on + np.nanargmax(y[i_on:search_end])
                    
#                     # Find offset (simplified)
#                     i_off = None
#                     for k in range(i_peak, min(T, i_peak + int(6.0 * fs))):
#                         if y[k] < th_off:
#                             i_off = k
#                             break
                    
#                     if i_off is not None:
#                         # FIXED: Store absolute times in seconds, not relative to onset
#                         event = {
#                             't_on': i_base/fs,     # Absolute time of baseline/onset
#                             't_peak': i_peak/fs,   # Absolute time of peak
#                             't_off': i_off/fs,     # Absolute time of offset
#                             'amp': amp, 
#                             'baseline': baseline,
#                             'i_on': i_on, 'i_peak': i_peak, 'i_off': i_off, 'i_base': i_base
#                         }
#                         events.append(event)
#                         mask[i_base:i_off] = True
#                         i = i_off + 1
#                         continue
#                     else:
#                         debug_info['rejected_reasons']['no_offset'] += 1
#                         debug_info['rejection_details'].append({
#                             'reason': 'no_offset',
#                             'i_on': i_on, 't_on': i_on/fs,
#                             'amp': amp, 'baseline': baseline
#                         })
#                 else:
#                     debug_info['rejected_reasons']['low_amp'] += 1
#                     debug_info['rejection_details'].append({
#                         'reason': 'low_amp',
#                         'i_on': i_on, 't_on': i_on/fs,
#                         'amp': amp, 'baseline': baseline,
#                         'required_amp': min_amp
#                     })
#             else:
#                 debug_info['rejected_reasons']['no_peak'] += 1
#                 debug_info['rejection_details'].append({
#                     'reason': 'no_peak',
#                     'i_on': i_on, 't_on': i_on/fs
#                 })
            
#             i += 1
#         else:
#             i += 1
    
#     debug_info['final_events'] = events
    
#     if debug:
#         print(f"\nCONCISE SUMMARY:")
#         print(f"  Found {onset_count} potential onsets -> {len(events)} final events")
#         print(f"  Rejection reasons:")
#         for reason, count in debug_info['rejected_reasons'].items():
#             print(f"    {reason}: {count}")
        
#         if len(events) > 0:
#             amps = [e['amp'] for e in events]
#             times = [e['t_peak'] for e in events]
#             print(f"  Event amplitudes: {np.min(amps):.3f} - {np.max(amps):.3f} ΔF/F")
#             print(f"  Event times: {np.min(times):.1f} - {np.max(times):.1f} s")
#             print(f"  Event rate: {len(events)/(T/fs):.3f} events/s")
            
#             # Show sample events with corrected timing
#             print(f"  Sample events (corrected timing):")
#             for i in range(min(5, len(events))):
#                 e = events[i]
#                 print(f"    Event {i+1}: t_on={e['t_on']:.2f}s, t_peak={e['t_peak']:.2f}s, t_off={e['t_off']:.2f}s, amp={e['amp']:.3f}")
    
#     return events, mask, debug_info


# def detect_ca_transients_debug_enhanced(x, fs, roi_id=None, debug=True, **params):
#     """
#     Enhanced debug version that returns threshold information and detailed rejection tracking
#     """
#     if debug and roi_id is not None:
#         print(f"\n=== DEBUGGING {roi_id} ===")
    
#     x = np.asarray(x, float)
#     T = x.size
    
#     if debug:
#         print(f"Input: {T} frames, fs={fs:.1f}Hz, duration={T/fs:.1f}s")
#         print(f"dF/F range: [{np.nanmin(x):.3f}, {np.nanmax(x):.3f}], std={np.nanstd(x):.3f}")
    
#     # Extract parameters from config (with your relaxed defaults)
#     smooth_sigma_s = params.get('smooth_sigma_s', 0.05)
#     k_on = params.get('k_on', 1.5)
#     k_off = params.get('k_off', 0.5)
#     min_amp = params.get('min_amp', 0.05)
#     min_series_peak_dff = params.get('min_series_peak_dff', 0.10)
    
#     if debug:
#         print(f"Config parameters: k_on={k_on}, k_off={k_off}, min_amp={min_amp}")
#         print(f"                  smooth_sigma_s={smooth_sigma_s}, min_series_peak_dff={min_series_peak_dff}")

#     if T < 2:
#         if debug: print("REJECT: Too few frames")
#         return [], np.zeros(T, dtype=bool), {'th_on': 0, 'th_off': 0, 'baseline': 0, 'sigma': 0}
    
#     # Check for invalid traces
#     if np.all(np.isnan(x)) or np.all(x == x[0]):
#         if debug: print("REJECT: Invalid trace (all NaN or constant)")
#         return [], np.zeros(T, dtype=bool), {'th_on': 0, 'th_off': 0, 'baseline': 0, 'sigma': 0}

#     # Smoothing (using config parameter)
#     if smooth_sigma_s > 0:
#         sigma_frames = max(1, int(round(smooth_sigma_s * fs)))
#         x_s = gaussian_filter1d(x, sigma=sigma_frames)
#         if debug: print(f"Smoothed with sigma={sigma_frames} frames")
#     else:
#         x_s = x.copy()

#     # Simple noise estimation (same as working version)
#     try:
#         valid_mask = np.isfinite(x_s)
#         if not np.any(valid_mask):
#             if debug: print("REJECT: No valid data points")
#             return [], np.zeros(T, dtype=bool), {'th_on': 0, 'th_off': 0, 'baseline': 0, 'sigma': 0}
        
#         x_valid = x_s[valid_mask]
        
#         if len(x_valid) < 10:
#             if debug: print("REJECT: Too few valid points")
#             return [], np.zeros(T, dtype=bool), {'th_on': 0, 'th_off': 0, 'baseline': 0, 'sigma': 0}
        
#         # Simple noise estimation (same as working version)
#         median_val = np.median(x_valid)
        
#         # Use MAD for noise estimate
#         mad = np.median(np.abs(x_valid - median_val))
#         if mad < 1e-12:
#             mad = np.std(x_valid) / 1.4826
#             if mad < 1e-12:
#                 if debug: print("REJECT: Zero variance trace")
#                 return [], np.zeros(T, dtype=bool), {'th_on': 0, 'th_off': 0, 'baseline': median_val, 'sigma': 0}
        
#         sigma = 1.4826 * mad
        
#         if debug:
#             print(f"Noise estimation: median={median_val:.3f}, sigma={sigma:.3f}, mad={mad:.3f}")
        
#     except Exception as e:
#         if debug: print(f"REJECT: Noise estimation error: {e}")
#         return [], np.zeros(T, dtype=bool), {'th_on': 0, 'th_off': 0, 'baseline': 0, 'sigma': 0}

#     # Thresholds (using config parameters)
#     y = x_s
#     th_on = median_val + k_on * sigma
#     th_off = median_val + k_off * sigma
    
#     if debug:
#         print(f"Thresholds: th_on={th_on:.3f}, th_off={th_off:.3f}")
#         above_on = np.sum(y > th_on)
#         above_off = np.sum(y > th_off)
#         print(f"Points above thresholds: {above_on} above th_on, {above_off} above th_off")

#     # Enhanced tracking of detection stages
#     debug_info = {
#         'th_on': th_on,
#         'th_off': th_off, 
#         'baseline': median_val,
#         'sigma': sigma,
#         'smoothed_trace': y.copy(),
#         'potential_onsets': [],
#         'rejected_reasons': {'no_peak': 0, 'low_amp': 0, 'no_offset': 0},
#         'rejection_details': [],  # Store details for each rejection
#         'final_events': []
#     }
    
#     events = []
#     mask = np.zeros(T, dtype=bool)
#     i = 1
#     onset_count = 0
    
#     # SIMPLE detection loop (same as working version) but using CONFIG PARAMETERS
#     while i < T:
#         if not np.isfinite(y[i-1]) or not np.isfinite(y[i]):
#             i += 1
#             continue
            
#         # ONSET detection
#         if y[i-1] < th_on and y[i] >= th_on:
#             onset_count += 1
#             i_on = i
            
#             # Store potential onset
#             debug_info['potential_onsets'].append({
#                 'i_on': i_on,
#                 't_on': i_on/fs,
#                 'value': y[i_on]
#             })
            
#             # Find baseline - look backwards from onset to find last time below th_off
#             j = i_on - 1  # Start just before onset
#             while j >= 0 and np.isfinite(y[j]) and y[j] > th_off:
#                 j -= 1
#             i_base = max(0, j)  # Don't go negative
#             baseline = y[i_base]
            
#             # Quick amplitude check
#             search_end = min(T, i_on + int(2.0 * fs))  # 2 second search
#             if search_end > i_on:
#                 peak_val = np.nanmax(y[i_on:search_end])
#                 amp = peak_val - baseline
                
#                 # Use config parameters for amplitude checks
#                 if amp >= min_amp and amp >= min_series_peak_dff:
#                     # For debugging, create a simple event
#                     i_peak = i_on + np.nanargmax(y[i_on:search_end])
                    
#                     # Find offset (simplified)
#                     i_off = None
#                     for k in range(i_peak, min(T, i_peak + int(6.0 * fs))):
#                         if y[k] < th_off:
#                             i_off = k
#                             break
                    
#                     if i_off is not None:
#                         # Store absolute times in seconds (FIXED from previous version)
#                         event = {
#                             't_on': i_base/fs,     # Absolute time of baseline/onset
#                             't_peak': i_peak/fs,   # Absolute time of peak
#                             't_off': i_off/fs,     # Absolute time of offset
#                             'amp': amp, 
#                             'baseline': baseline,
#                             'i_on': i_on, 'i_peak': i_peak, 'i_off': i_off, 'i_base': i_base
#                         }
#                         events.append(event)
#                         mask[i_base:i_off] = True
#                         i = i_off + 1
#                         continue
#                     else:
#                         debug_info['rejected_reasons']['no_offset'] += 1
#                         debug_info['rejection_details'].append({
#                             'reason': 'no_offset',
#                             'i_on': i_on, 't_on': i_on/fs,
#                             'amp': amp, 'baseline': baseline
#                         })
#                 else:
#                     debug_info['rejected_reasons']['low_amp'] += 1
#                     debug_info['rejection_details'].append({
#                         'reason': 'low_amp',
#                         'i_on': i_on, 't_on': i_on/fs,
#                         'amp': amp, 'baseline': baseline,
#                         'required_min_amp': min_amp,
#                         'required_series_peak': min_series_peak_dff
#                     })
#             else:
#                 debug_info['rejected_reasons']['no_peak'] += 1
#                 debug_info['rejection_details'].append({
#                     'reason': 'no_peak',
#                     'i_on': i_on, 't_on': i_on/fs
#                 })
            
#             i += 1
#         else:
#             i += 1
    
#     debug_info['final_events'] = events
    
#     if debug:
#         print(f"\nCONCISE SUMMARY:")
#         print(f"  Found {onset_count} potential onsets -> {len(events)} final events")
#         print(f"  Rejection reasons:")
#         for reason, count in debug_info['rejected_reasons'].items():
#             if count > 0:
#                 print(f"    {reason}: {count}")
        
#         if len(events) > 0:
#             amps = [e['amp'] for e in events]
#             times = [e['t_peak'] for e in events]
#             print(f"  Event amplitudes: {np.min(amps):.3f} - {np.max(amps):.3f} ΔF/F")
#             print(f"  Event times: {np.min(times):.1f} - {np.max(times):.1f} s")
#             print(f"  Event rate: {len(events)/(T/fs):.3f} events/s")
            
#             # Show sample events with corrected timing
#             print(f"  Sample events:")
#             for i in range(min(3, len(events))):
#                 e = events[i]
#                 print(f"    Event {i+1}: t_on={e['t_on']:.2f}s, t_peak={e['t_peak']:.2f}s, t_off={e['t_off']:.2f}s, amp={e['amp']:.3f}")
#         else:
#             print(f"  No events detected with current config parameters")
#             # Show a few rejection examples
#             if debug_info['rejection_details']:
#                 print(f"  Sample rejections:")
#                 for i, rejection in enumerate(debug_info['rejection_details'][:3]):
#                     print(f"    Rejection {i+1}: {rejection['reason']} at t={rejection['t_on']:.2f}s")
#                     if 'amp' in rejection:
#                         print(f"                     amp={rejection['amp']:.3f}, required={rejection.get('required_min_amp', 'N/A')}")
    
#     return events, mask, debug_info



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
# STEP 7: Fluorescence processing
data = neuropil_regress(data, cfg)


# %%
data = compute_baseline_and_dff(data, cfg)



# %%
compute_qc_metrics(data, cfg)


# %%


# Get QC metrics
qc = data['qc_metrics']

# Total events per ROI
event_counts = qc['ca_event_counts']  # Shape: (N_rois,)

# For specific ROI
roi = 18
for roi in range(30):
    total_events_roi = qc['ca_event_counts'][roi]
    actual_events_roi = qc['ca_events'][roi]  # List of event dicts

    print(f"ROI {roi}: {total_events_roi} events")
    # print(f"Event times: {[e['t_peak'] for e in actual_events_roi]}")

# %%


# Get QC metrics
qc = data['qc_metrics']

# Total events per ROI
event_counts = qc['ca_event_counts']  # Shape: (N_rois,)


for roi in range(data['dFF'].shape[0]):
    total_events_roi = qc['ca_event_counts'][roi]
    actual_events_roi = qc['ca_events'][roi]  # List of event dicts

    print(f"ROI {roi}: {total_events_roi} events")

# %%

# Get QC metrics
qc = data['qc_metrics']

# Total events per ROI
event_counts = qc['ca_event_counts']  # Shape: (N_rois,)


for roi in range(445,485):
    total_events_roi = qc['ca_event_counts'][roi]
    actual_events_roi = qc['ca_events'][roi]  # List of event dicts

    print(f"ROI {roi}: {total_events_roi} events")




# %%
# Debug Ca transient detection on specific ROIs
compute_qc_metrics_debug(data, cfg, debug_rois=[2, 18, 21])

# Or test just one ROI interactively
roi = 18
events, mask, debug_info = detect_ca_transients_debug(
    data['dFF_clean'][roi], cfg['acq']['fs'], 
    roi_id=roi, debug=True, **cfg['ca_transients']
)


# %%
cfg = load_cfg_yaml(cfg_path)
# Enhanced debug visualization for ROI 18
debug_ca_detection_with_visualization(data, cfg, roi=18, save=True)

# Also try ROI 2 to see the difference
debug_ca_detection_with_visualization(data, cfg, roi=2, save=True)


debug_ca_detection_with_visualization(data, cfg, roi=30, save=True)


# %%



for roi in range(30):
    debug_ca_detection_with_visualization(data, cfg, roi=roi, save=True)


# %%
# PARAMETER SWEEP WITH DEBUG VISUALIZATION
print("\n" + "="*50)
print("RUNNING PARAMETER SWEEP WITH DEBUG VISUALIZATION")
print("="*50)

# Define parameter combinations to test
param_sweeps = [
    {
        'name': 'Current (permissive)',
        'params': {
            'k_on': 2.0, 'k_off': 1.0, 'min_amp': 0.12, 
            'min_series_peak_dff': 0.20, 'min_snr': 2.0
        }
    },
    {
        'name': 'Conservative',
        'params': {
            'k_on': 2.8, 'k_off': 1.5, 'min_amp': 0.15,
            'min_series_peak_dff': 0.25, 'min_snr': 2.5
        }
    },
    {
        'name': 'Very Conservative',
        'params': {
            'k_on': 3.0, 'k_off': 1.8, 'min_amp': 0.18,
            'min_series_peak_dff': 0.30, 'min_snr': 3.0
        }
    },
    {
        'name': 'Moderate',
        'params': {
            'k_on': 2.5, 'k_off': 1.2, 'min_amp': 0.13,
            'min_series_peak_dff': 0.22, 'min_snr': 2.2
        }
    }
]

# ROI selection - mix of good, noisy, and intermediate
test_rois = [2, 10, 18, 21, 25, 30, 35, 40]  # Adjust based on your data

print(f"Testing {len(param_sweeps)} parameter sets on ROIs: {test_rois}")

# Store results for comparison
sweep_results = {}

for sweep in param_sweeps:
    sweep_name = sweep['name']
    sweep_params = sweep['params']
    
    print(f"\n{'='*30}")
    print(f"TESTING: {sweep_name}")
    print(f"Parameters: {sweep_params}")
    print(f"{'='*30}")
    
    # Update config with sweep parameters
    cfg_test = cfg.copy()
    cfg_test['ca_transients'].update(sweep_params)
    
    roi_results = {}
    
    for roi in test_rois:
        if roi >= data['dFF'].shape[0]:
            continue
            
        print(f"\n--- ROI {roi} with {sweep_name} ---")
        
        try:
            # Run debug detection with current parameters
            events, mask, debug_info = detect_ca_transients_debug_enhanced(
                data['dFF_clean'][roi], 
                cfg['acq']['fs'], 
                roi_id=f"{roi}_{sweep_name}", 
                debug=False,  # Reduced output for sweep
                **cfg_test['ca_transients']
            )
            
            # Store key metrics
            roi_results[roi] = {
                'n_events': len(events),
                'event_rate': len(events) / (data['dFF'].shape[1] / cfg['acq']['fs']),
                'mean_amp': np.mean([e['amp'] for e in events]) if events else 0,
                'total_rejected': sum(debug_info['rejected_reasons'].values()),
                'rejection_breakdown': debug_info['rejected_reasons'].copy()
            }
            
            print(f"  Events: {len(events)}, Rate: {roi_results[roi]['event_rate']:.2f}/s, Mean amp: {roi_results[roi]['mean_amp']:.3f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            roi_results[roi] = {'n_events': -1, 'error': str(e)}
    
    sweep_results[sweep_name] = roi_results

# Summary comparison
print(f"\n{'='*50}")
print("PARAMETER SWEEP SUMMARY")
print(f"{'='*50}")

print(f"{'ROI':<5} | {'Current':<10} | {'Conservative':<12} | {'V.Conserv':<10} | {'Moderate':<10}")
print("-" * 65)

for roi in test_rois:
    if roi >= data['dFF'].shape[0]:
        continue
    
    row = f"{roi:<5} | "
    for sweep_name in ['Current (permissive)', 'Conservative', 'Very Conservative', 'Moderate']:
        if sweep_name in sweep_results and roi in sweep_results[sweep_name]:
            n_events = sweep_results[sweep_name][roi]['n_events']
            if n_events >= 0:
                row += f"{n_events:<10} | "
            else:
                row += f"{'ERROR':<10} | "
        else:
            row += f"{'N/A':<10} | "
    print(row)

# Detailed breakdown for each parameter set
for sweep_name, roi_data in sweep_results.items():
    print(f"\n{sweep_name.upper()} - DETAILED BREAKDOWN:")
    
    all_events = [data['n_events'] for data in roi_data.values() if data['n_events'] >= 0]
    if all_events:
        print(f"  Total events across ROIs: {sum(all_events)}")
        print(f"  Average events per ROI: {np.mean(all_events):.1f}")
        print(f"  ROIs with events: {sum(1 for n in all_events if n > 0)}/{len(all_events)}")
        print(f"  Event range: {min(all_events)} - {max(all_events)}")
        
        # Show which ROIs have many vs few events
        high_event_rois = [roi for roi, data in roi_data.items() 
                          if data.get('n_events', 0) > 10]
        low_event_rois = [roi for roi, data in roi_data.items() 
                         if 0 <= data.get('n_events', -1) <= 2]
        
        print(f"  High activity ROIs (>10 events): {high_event_rois}")
        print(f"  Low activity ROIs (≤2 events): {low_event_rois}")

# Recommendation
print(f"\n{'='*50}")
print("RECOMMENDATION")
print(f"{'='*50}")

print("Based on the sweep results:")
print("1. Look for parameter set that gives good separation between good/noisy ROIs")
print("2. Good ROIs (like 18) should have reasonable event counts (5-50)")
print("3. Noisy ROIs (like 10) should have very few events (0-2)")
print("4. Consider the 'Conservative' or 'Moderate' settings for trace quality filtering")

# Optional: Generate debug plots for best parameter set
best_params = 'Conservative'  # Adjust based on results
print(f"\nGenerating debug plots for recommended parameters: {best_params}")

cfg_final = cfg.copy()
cfg_final['ca_transients'].update(param_sweeps[1]['params'])  # Conservative

# Quick visual check on a few representative ROIs
for roi in [10, 18]:  # Noisy vs good
    if roi < data['dFF'].shape[0]:
        print(f"\nGenerating debug plot for ROI {roi} with {best_params} parameters...")
        debug_ca_detection_with_visualization(data, cfg_final, roi=roi, save=True)

print(f"\nParameter sweep complete! Check the debug plots to verify the separation.")

# %%
# ENHANCED ROI DEBUG ANALYSIS WITH SUMMARY INFO
print("\n" + "="*60)
print("ENHANCED ROI DEBUG ANALYSIS")
print("="*60)

# Define ROIs to analyze - mix of different types
debug_roi_list = [2, 10, 18, 21, 25, 30, 35, 40, 445, 450, 455, 460]  # Customize this list

# Use current moderate parameters (or adjust as needed)
debug_params = {
    'k_on': 2.5, 'k_off': 1.2, 'min_amp': 0.13,
    'min_series_peak_dff': 0.22, 'min_snr': 2.2
}

debug_params_tight = {
    'k_on': 2.8, 'k_off': 1.2, 'min_amp': 0.13,
    'min_series_peak_dff': 0.22, 'min_snr': 3.0,  # Increased from 2.2
    'rise_max_s': 0.08  # Keep this - it's working well
}


refined_params = {
    'k_on': 2.8,           # Slight increase to reduce false positives
    'k_off': 1.2,          # Keep same
    'min_amp': 0.14,       # Slight increase from 0.13
    'min_series_peak_dff': 0.24,  # Slight increase from 0.22
    'min_snr': 2.5,        # Increase from 2.2 to filter more noise
    'rise_max_s': 0.08,    # Keep tight - working very well
    'decay_max_s': 4.0,
    'require_clear_peak': True
}


moderate_params_proven = {
    'k_on': 2.5,           # REVERT - this was working
    'k_off': 1.2,          # Keep same
    'min_amp': 0.13,       # REVERT - was good
    'min_series_peak_dff': 0.22,  # REVERT - was working well  
    'min_snr': 2.2,        # REVERT - 2.5 was too strict
    'rise_max_s': 0.08,    # Keep - this filter works well
    'decay_max_s': 4.0,
    'require_clear_peak': False  # DISABLE - this killed everything
}

moderate_params_proven = {
    'k_on': 2.5,           # REVERT - this was working
    'k_off': 2.3,          # Keep same
    'min_amp': 0.13,       # REVERT - was good
    'min_series_peak_dff': 0.22,  # REVERT - was working well  
    'min_snr': 0.2,        # REVERT - 2.5 was too strict
    'rise_max_s': 0.2,    # Keep - this filter works well
    'decay_max_s': 2.0,
    'require_clear_peak': False  # DISABLE - this killed everything
}

debug_params = moderate_params_proven

# Update config for debug run
cfg_debug = cfg.copy()
cfg_debug['ca_transients'].update(debug_params)

print(f"Debug parameters: {debug_params}")
print(f"Analyzing ROIs: {debug_roi_list}")
print(f"Total ROIs to process: {len(debug_roi_list)}")

# Storage for summary data
debug_summary = {
    'roi_id': [],
    'roi_label': [],
    'n_events': [],
    'event_rate': [],
    'mean_amplitude': [],
    'total_rejected': [],
    'rejection_breakdown': [],
    'trace_stats': [],
    'area_um2': [],
    'aspect_ratio': [],
    'coherence': []
}

# Process each ROI
for i, roi in enumerate(debug_roi_list):
    if roi >= data['dFF'].shape[0]:
        print(f"\nROI {roi} out of range (max: {data['dFF'].shape[0]-1}), skipping...")
        continue
    
    print(f"\n{'='*40}")
    print(f"PROCESSING ROI {roi} ({i+1}/{len(debug_roi_list)})")
    print(f"{'='*40}")
    
    # Get ROI metadata
    roi_label = data['roi_labels'][roi] if roi < len(data['roi_labels']) else 'unknown'
    feats = data['roi_features']
    area_um2 = feats['area_um2'][roi] if roi < len(feats['area_um2']) else 0
    aspect_ratio = feats['aspect_ratio'][roi] if roi < len(feats['aspect_ratio']) else 0
    coherence = feats.get('orientation_coherence', [0]*len(feats['area_um2']))[roi] if roi < len(feats.get('orientation_coherence', [])) else 0
    
    print(f"ROI {roi} metadata:")
    print(f"  Label: {roi_label}")
    print(f"  Area: {area_um2:.1f} µm²")
    print(f"  Aspect ratio: {aspect_ratio:.2f}")
    print(f"  Coherence: {coherence:.3f}")
    
    # Analyze trace statistics
    trace = data['dFF_clean'][roi]
    trace_valid = trace[np.isfinite(trace)]
    
    trace_stats = {
        'mean': np.mean(trace_valid),
        'std': np.std(trace_valid),
        'min': np.min(trace_valid),
        'max': np.max(trace_valid),
        'range': np.max(trace_valid) - np.min(trace_valid),
        'cv': np.std(trace_valid) / (np.abs(np.mean(trace_valid)) + 1e-9)
    }
    
    print(f"  Trace stats: mean={trace_stats['mean']:.3f}, std={trace_stats['std']:.3f}")
    print(f"               range=[{trace_stats['min']:.3f}, {trace_stats['max']:.3f}], CV={trace_stats['cv']:.2f}")
    
    # Run enhanced debug detection on full trace
    try:
        events, mask, debug_info = detect_ca_transients_debug_enhanced(
            trace, cfg['acq']['fs'], 
            roi_id=f"ROI_{roi}", 
            debug=True, 
            **cfg_debug['ca_transients']
        )
        
        # Calculate summary metrics
        n_events = len(events)
        event_rate = n_events / (len(trace) / cfg['acq']['fs'])
        mean_amp = np.mean([e['amp'] for e in events]) if events else 0
        total_rejected = sum(debug_info['rejected_reasons'].values())
        
        print(f"\nDetection summary:")
        print(f"  Events detected: {n_events}")
        print(f"  Event rate: {event_rate:.3f} events/s")
        print(f"  Mean amplitude: {mean_amp:.3f} ΔF/F")
        print(f"  Total rejected: {total_rejected}")
        print(f"  Rejection breakdown: {debug_info['rejected_reasons']}")
        
        # Store summary data
        debug_summary['roi_id'].append(roi)
        debug_summary['roi_label'].append(roi_label)
        debug_summary['n_events'].append(n_events)
        debug_summary['event_rate'].append(event_rate)
        debug_summary['mean_amplitude'].append(mean_amp)
        debug_summary['total_rejected'].append(total_rejected)
        debug_summary['rejection_breakdown'].append(debug_info['rejected_reasons'].copy())
        debug_summary['trace_stats'].append(trace_stats)
        debug_summary['area_um2'].append(area_um2)
        debug_summary['aspect_ratio'].append(aspect_ratio)
        debug_summary['coherence'].append(coherence)
        
    except Exception as e:
        print(f"ERROR in detection: {e}")
        # Store error data
        debug_summary['roi_id'].append(roi)
        debug_summary['roi_label'].append(roi_label)
        debug_summary['n_events'].append(-1)
        debug_summary['event_rate'].append(0)
        debug_summary['mean_amplitude'].append(0)
        debug_summary['total_rejected'].append(0)
        debug_summary['rejection_breakdown'].append({})
        debug_summary['trace_stats'].append(trace_stats)
        debug_summary['area_um2'].append(area_um2)
        debug_summary['aspect_ratio'].append(aspect_ratio)
        debug_summary['coherence'].append(coherence)
    
    # Generate debug visualization with enhanced title
    try:
        print(f"\nGenerating debug visualization for ROI {roi}...")
        
        # Temporarily modify the debug function to include ROI in title
        # We'll need to pass this info through somehow
        debug_ca_detection_with_visualization(data, cfg_debug, roi=roi, save=True)
        
    except Exception as e:
        print(f"ERROR in visualization: {e}")



# Add this section after the main processing loop:
print(f"\nDETAILED PER-ROI BREAKDOWN:")
for i, roi_id in enumerate(debug_summary['roi_id']):
    roi = debug_summary['roi_id'][i]
    events = debug_summary['n_events'][i]
    rejections = debug_summary['rejection_breakdown'][i]
    
    print(f"\nROI {roi} ({debug_summary['roi_label'][i]}):")
    print(f"  Events: {events}")
    print(f"  Top rejections: {dict(sorted(rejections.items(), key=lambda x: x[1], reverse=True)[:3])}")
    
    # Trace quality indicators
    trace_stats = debug_summary['trace_stats'][i]
    print(f"  Trace quality: std={trace_stats['std']:.3f}, CV={trace_stats['cv']:.2f}, range={trace_stats['range']:.3f}")
    
    # Classification suggestion
    if events > 200:
        classification = "HIGH ACTIVITY (good Ca²⁺)"
    elif events > 50:
        classification = "MODERATE ACTIVITY"
    elif events > 10:
        classification = "LOW ACTIVITY (borderline)"
    else:
        classification = "MINIMAL ACTIVITY (likely noise)"
    
    print(f"  Suggested classification: {classification}")



# Print comprehensive summary
print(f"\n{'='*60}")
print("COMPREHENSIVE DEBUG SUMMARY")
print(f"{'='*60}")

print(f"\nOverall statistics:")
successful_rois = [i for i, n in enumerate(debug_summary['n_events']) if n >= 0]
print(f"  Successfully processed: {len(successful_rois)}/{len(debug_summary['roi_id'])} ROIs")

if len(successful_rois) > 0:
    event_counts = [debug_summary['n_events'][i] for i in successful_rois]
    event_rates = [debug_summary['event_rate'][i] for i in successful_rois]
    
    print(f"  Event count range: {min(event_counts)} - {max(event_counts)}")
    print(f"  Mean events per ROI: {np.mean(event_counts):.1f}")
    print(f"  Event rate range: {min(event_rates):.3f} - {max(event_rates):.3f} events/s")
    
    # Breakdown by ROI label
    print(f"\nBreakdown by ROI classification:")
    for label in ['soma', 'process', 'uncertain']:
        label_indices = [i for i in successful_rois if debug_summary['roi_label'][i] == label]
        if len(label_indices) > 0:
            label_events = [debug_summary['n_events'][i] for i in label_indices]
            label_rates = [debug_summary['event_rate'][i] for i in label_indices]
            print(f"  {label.upper()} ROIs (n={len(label_indices)}):")
            print(f"    Event count: {np.mean(label_events):.1f} ± {np.std(label_events):.1f} (range: {min(label_events)}-{max(label_events)})")
            print(f"    Event rate: {np.mean(label_rates):.3f} ± {np.std(label_rates):.3f} events/s")

# Create summary table
print(f"\nDETAILED ROI TABLE:")
print(f"{'ROI':<5} | {'Label':<10} | {'Events':<7} | {'Rate':<8} | {'MeanAmp':<8} | {'Rejected':<9} | {'Area_µm²':<8} | {'AR':<5} | {'Coh':<5}")
print("-" * 85)

for i, roi_id in enumerate(debug_summary['roi_id']):
    roi = debug_summary['roi_id'][i]
    label = debug_summary['roi_label'][i][:9]  # Truncate long labels
    events = debug_summary['n_events'][i]
    rate = debug_summary['event_rate'][i]
    amp = debug_summary['mean_amplitude'][i]
    rejected = debug_summary['total_rejected'][i]
    area = debug_summary['area_um2'][i]
    ar = debug_summary['aspect_ratio'][i]
    coh = debug_summary['coherence'][i]
    
    print(f"{roi:<5} | {label:<10} | {events:<7} | {rate:<8.3f} | {amp:<8.3f} | {rejected:<9} | {area:<8.1f} | {ar:<5.2f} | {coh:<5.3f}")

# Rejection analysis
print(f"\nREJECTION ANALYSIS:")
all_rejections = {}
for breakdown in debug_summary['rejection_breakdown']:
    for reason, count in breakdown.items():
        all_rejections[reason] = all_rejections.get(reason, 0) + count

if all_rejections:
    total_rejections = sum(all_rejections.values())
    print(f"Total rejections across all ROIs: {total_rejections}")
    print(f"Rejection reasons:")
    for reason, count in sorted(all_rejections.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = 100 * count / total_rejections
            print(f"  {reason}: {count} ({pct:.1f}%)")

# Recommendations
print(f"\nRECOMMENDations:")
high_event_rois = [debug_summary['roi_id'][i] for i in successful_rois if debug_summary['n_events'][i] > 5]
low_event_rois = [debug_summary['roi_id'][i] for i in successful_rois if debug_summary['n_events'][i] <= 1]

print(f"  High activity ROIs (>5 events): {high_event_rois}")
print(f"  Low activity ROIs (≤1 event): {low_event_rois}")
print(f"  Good separation: {len(high_event_rois)} high vs {len(low_event_rois)} low")

if len(all_rejections) > 0:
    top_rejection = max(all_rejections, key=all_rejections.get)
    print(f"  Main rejection reason: {top_rejection} ({all_rejections[top_rejection]} cases)")
    
    if top_rejection == 'low_amp':
        print(f"    → Consider lowering min_amp or min_series_peak_dff")
    elif top_rejection == 'no_offset':
        print(f"    → Consider increasing decay_max_s or lowering k_off")
    elif top_rejection == 'low_snr':
        print(f"    → Consider lowering min_snr threshold")

print(f"\nDebug analysis complete! Check the generated plots for visual verification.")
print(f"Save directory: {cfg['overlay']['save_dir']}")






# %%
# Run the timing diagnostic
debug_ca_detection_timing_issue(data, cfg, roi=18)


# %%
# STEP 8: Summary
summarize(data, cfg)
quick_plot(data, cfg, roi=0)



# %%
# STEP 6: Detailed review plots
prepare_review_cache(data, cfg)
_ = plot_roi_review(data, cfg, roi=0, save=False)

# %%
# STEP 7: Batch export (optional)
# batch_plot_rois(data, cfg, list(range(data['dFF'].shape[0])), limit=30)
# batch_plot_rois(data, cfg, [2,18], limit=30)
batch_plot_rois(data, cfg, list(range(445,485)), limit=None)


print("\n" + "="*50)
print("PIPELINE COMPLETE")
print("="*50)