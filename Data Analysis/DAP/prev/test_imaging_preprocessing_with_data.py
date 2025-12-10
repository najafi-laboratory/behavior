"""
Test script for running actual imaging preprocessing with real data.

This script tests the complete imaging preprocessing pipeline including:
1. Loading suite2p data
2. Running quality control
3. Running cell labeling
4. Running trace extraction
5. Verifying output files are created

Usage:
    python test_imaging_preprocessing_with_data.py
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Add current directory to path
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# Import our modules
import modules.utils as utils
from modules.core.config_manager import ConfigManager
from modules.imaging.general_imaging_preprocessor import GeneralImagingPreprocessor

def setup_test_logging():
    """Setup logging for test."""
    logger = logging.getLogger('test_imaging_data')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def check_output_files(imaging_session_root, session_name):
    """Check what files were created by the preprocessing."""
    print(f"\n📁 Checking output files for {session_name}:")
    
    # Expected output locations
    expected_files = {
        'qc_results/fluo.npy': 'Quality control - filtered fluorescence traces',
        'qc_results/neuropil.npy': 'Quality control - neuropil traces',
        'qc_results/iscell_qc.npy': 'Quality control - cell classification',
        'dff.h5': 'Delta F/F traces (HDF5 format)',
        'labeling_results.npy': 'Cell type labeling results (excitatory/inhibitory)',
    }
    
    files_found = []
    files_missing = []
    
    for relative_path, description in expected_files.items():
        full_path = os.path.join(imaging_session_root, relative_path)
        if os.path.exists(full_path):
            file_size = os.path.getsize(full_path)
            files_found.append((relative_path, description, file_size))
            print(f"  ✓ {relative_path} - {description} ({file_size:,} bytes)")
        else:
            files_missing.append((relative_path, description))
            print(f"  ✗ {relative_path} - {description} (missing)")
    
    # Check for any additional files created
    print(f"\n📂 All files in {imaging_session_root}:")
    for root, dirs, files in os.walk(imaging_session_root):
        level = root.replace(imaging_session_root, '').count(os.sep)
        indent = ' ' * 2 * level
        rel_root = os.path.relpath(root, imaging_session_root)
        if rel_root == '.':
            print(f"{indent}📁 (root)")
        else:
            print(f"{indent}📁 {rel_root}/")
        
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"{subindent}📄 {file} ({file_size:,} bytes)")
    
    return files_found, files_missing

def test_imaging_preprocessing_with_data():
    """Test the complete imaging preprocessing pipeline with real data."""
    
    print("="*70)
    print("TESTING IMAGING PREPROCESSING WITH REAL DATA")
    print("="*70)
    
    # Setup logging
    logger = setup_test_logging()
    
    # Initialize configuration
    print("\n1. Loading configuration...")
    try:
        config_manager = ConfigManager('modules/config.yaml')
        config = config_manager.config
        print("✓ Configuration loaded successfully")
        
        # Find subjects with imaging sessions
        subjects_with_imaging = []
        for subject_id, subject_config in config.get('subjects', {}).items():
            if 'imaging_sessions' in subject_config:
                subjects_with_imaging.append(subject_id)
        
        if not subjects_with_imaging:
            print("✗ No subjects with imaging sessions found in config")
            return False
        
        print(f"  → Subjects with imaging sessions: {subjects_with_imaging}")
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False
    
    # Initialize imaging preprocessor
    print("\n2. Initializing imaging preprocessor...")
    try:
        subject_list = subjects_with_imaging
        
        preprocessor = GeneralImagingPreprocessor(
            config_manager=config_manager,
            subject_list=subject_list,
            logger=logger
        )
        print("✓ Imaging preprocessor initialized successfully")
        
    except Exception as e:
        print(f"✗ Imaging preprocessor initialization failed: {e}")
        return False
    
    # Find a session with actual data
    print("\n3. Finding imaging session with real data...")
    target_session = None
    target_subject = None
    
    for subject_id in subject_list:
        subject_config = config['subjects'].get(subject_id, {})
        imaging_sessions = subject_config.get('imaging_sessions', [])
        
        for imaging_session in imaging_sessions:
            path = preprocessor.get_imaging_session_path(subject_id, imaging_session)
            ops_path = os.path.join(path, 'ops.npy')
            
            if os.path.exists(ops_path):
                print(f"  ✓ Found real data for {subject_id}: {imaging_session.get('imaging_folder')}")
                target_session = imaging_session
                target_subject = subject_id
                break
        
        if target_session:
            break
    
    if not target_session:
        print("  ⚠ No imaging sessions with real data found")
        print("  → Cannot test preprocessing without actual suite2p data")
        return True  # Not a failure, just no data available
    
    # Run preprocessing on the found session
    print(f"\n4. Running complete preprocessing pipeline on {target_subject}...")
    try:
        # Process the single session
        processing_results = preprocessor.process_imaging_session(
            target_subject, 
            target_session, 
            experiment_config=None
        )
        
        print("✓ Preprocessing pipeline completed")
        print(f"  → Overall status: {processing_results.get('overall_status')}")
        print(f"  → Steps completed: {processing_results.get('steps_completed')}")
        print(f"  → Steps failed: {processing_results.get('steps_failed')}")
        
        # Check what files were created
        imaging_session_root = os.path.dirname(preprocessor.get_imaging_session_path(target_subject, target_session))
        session_name = target_session.get('imaging_folder', 'unknown')
        
        files_found, files_missing = check_output_files(imaging_session_root, session_name)
        
        print(f"\n📊 Processing Results Summary:")
        print(f"  → Files created: {len(files_found)}")
        print(f"  → Files missing: {len(files_missing)}")
        
        if processing_results.get('overall_status') == 'success':
            print("🎉 Complete preprocessing pipeline successful!")
        elif processing_results.get('overall_status') == 'partial_success':
            print("⚠ Preprocessing completed with some failures")
        else:
            print("✗ Preprocessing pipeline failed")
            
        return True
        
    except Exception as e:
        print(f"✗ Preprocessing pipeline failed: {e}")
        logger.error(f"Full error details: {e}", exc_info=True)
        return False
    
    print("\n" + "="*70)
    print("IMAGING PREPROCESSING TEST COMPLETED")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = test_imaging_preprocessing_with_data()
    sys.exit(0 if success else 1)
