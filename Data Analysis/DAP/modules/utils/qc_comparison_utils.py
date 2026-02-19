"""
QC Comparison Utilities

Utility functions to compare Suite2p QC outputs between original standalone 
processing and the new pipeline processing to ensure identical results.
"""

import os
import numpy as np
import h5py
import logging
from typing import Dict, Any, List, Tuple, Optional


class QCOutputComparator:
    """
    Compare QC outputs between original and pipeline processing.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the QC output comparator.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def compare_npy_files(self, file1_path: str, file2_path: str, filename: str) -> Dict[str, Any]:
        """
        Compare two .npy files and return comparison results.
        
        Args:
            file1_path: Path to first .npy file (original)
            file2_path: Path to second .npy file (pipeline)
            filename: Name of file being compared for logging
            
        Returns:
            Dictionary with comparison results
        """
        result = {
            'filename': filename,
            'identical': False,
            'both_exist': False,
            'error': None,
            'details': {}
        }
        
        try:
            # Check if both files exist
            if not os.path.exists(file1_path):
                result['error'] = f"Original file missing: {file1_path}"
                return result
            
            if not os.path.exists(file2_path):
                result['error'] = f"Pipeline file missing: {file2_path}"
                return result
            
            result['both_exist'] = True
            
            # Load the files
            data1 = np.load(file1_path, allow_pickle=True)
            data2 = np.load(file2_path, allow_pickle=True)
            
            # Compare basic properties
            result['details']['shape1'] = data1.shape if hasattr(data1, 'shape') else 'scalar'
            result['details']['shape2'] = data2.shape if hasattr(data2, 'shape') else 'scalar'
            result['details']['dtype1'] = str(data1.dtype) if hasattr(data1, 'dtype') else str(type(data1))
            result['details']['dtype2'] = str(data2.dtype) if hasattr(data2, 'dtype') else str(type(data2))
            
            # Handle different data types
            if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
                # Check if this is a 0-dimensional array (single object like ops.npy)
                if data1.ndim == 0 and data2.ndim == 0:
                    # Handle 0-dimensional arrays (like ops.npy containing a single dict)
                    try:
                        obj1 = data1.item()
                        obj2 = data2.item()
                        if isinstance(obj1, dict) and isinstance(obj2, dict):
                            result['identical'] = self._compare_dicts(obj1, obj2)
                            # Add detailed comparison info for dict differences
                            if not result['identical']:
                                result['details']['dict_comparison'] = self._get_dict_differences(obj1, obj2)
                        else:
                            result['identical'] = (obj1 == obj2)
                    except Exception as e:
                        result['error'] = f"0-dimensional array comparison failed: {str(e)}"
                        return result
                # Check if this is a regular numeric array or object array
                elif data1.dtype == 'object' or data2.dtype == 'object':
                    # Handle object arrays (like stat.npy) specially
                    try:
                        result['identical'] = self._compare_object_arrays(data1, data2)
                    except Exception as e:
                        result['error'] = f"Object array comparison failed: {str(e)}"
                        return result
                else:
                    # Regular numpy arrays
                    result['identical'] = np.array_equal(data1, data2)
                    if not result['identical'] and data1.shape == data2.shape:
                        # Calculate difference metrics for numeric arrays
                        if np.issubdtype(data1.dtype, np.number) and np.issubdtype(data2.dtype, np.number):
                            diff = np.abs(data1 - data2)
                            result['details']['max_diff'] = np.max(diff)
                            result['details']['mean_diff'] = np.mean(diff)
                            result['details']['std_diff'] = np.std(diff)
                            
                            # Check if differences are within tolerance
                            tolerance = 1e-10
                            result['details']['within_tolerance'] = np.all(diff < tolerance)
                        
            elif hasattr(data1, '__len__') and hasattr(data2, '__len__'):
                # Object arrays or lists (like stat arrays)
                if len(data1) == len(data2):
                    result['identical'] = True
                    for i in range(len(data1)):
                        if data1[i] is None and data2[i] is None:
                            continue
                        elif data1[i] is None or data2[i] is None:
                            result['identical'] = False
                            break
                        elif isinstance(data1[i], dict) and isinstance(data2[i], dict):
                            # Compare dictionaries (like stat entries)
                            if not self._compare_dicts(data1[i], data2[i]):
                                result['identical'] = False
                                break
                        elif hasattr(data1[i], '__array__') and hasattr(data2[i], '__array__'):
                            # Compare array-like objects
                            try:
                                if not np.array_equal(np.asarray(data1[i]), np.asarray(data2[i])):
                                    result['identical'] = False
                                    break
                            except (ValueError, TypeError):
                                # If arrays can't be compared directly, convert to string
                                if str(data1[i]) != str(data2[i]):
                                    result['identical'] = False
                                    break
                        else:
                            # Direct comparison for other types
                            try:
                                # Use np.array_equal for safe comparison
                                if not np.array_equal(np.asarray(data1[i]), np.asarray(data2[i])):
                                    result['identical'] = False
                                    break
                            except (ValueError, TypeError):
                                # Fallback to string comparison
                                if str(data1[i]) != str(data2[i]):
                                    result['identical'] = False
                                    break
                else:
                    result['identical'] = False
            else:
                # Scalar values or other types
                result['identical'] = np.array_equal(data1, data2)
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"QC_COMP: Error comparing {filename}: {e}")
        
        return result
    
    def compare_h5_files(self, file1_path: str, file2_path: str, filename: str) -> Dict[str, Any]:
        """
        Compare two .h5 files (motion correction offsets).
        
        Args:
            file1_path: Path to first .h5 file (original)
            file2_path: Path to second .h5 file (pipeline)
            filename: Name of file being compared for logging
            
        Returns:
            Dictionary with comparison results
        """
        result = {
            'filename': filename,
            'identical': False,
            'both_exist': False,
            'error': None,
            'details': {}
        }
        
        try:
            # Check if both files exist
            if not os.path.exists(file1_path):
                result['error'] = f"Original file missing: {file1_path}"
                return result
            
            if not os.path.exists(file2_path):
                result['error'] = f"Pipeline file missing: {file2_path}"
                return result
            
            result['both_exist'] = True
            
            # Load and compare h5 files
            with h5py.File(file1_path, 'r') as f1, h5py.File(file2_path, 'r') as f2:
                # Check if both have same keys
                keys1 = set(f1.keys())
                keys2 = set(f2.keys())
                
                result['details']['keys1'] = list(keys1)
                result['details']['keys2'] = list(keys2)
                
                if keys1 != keys2:
                    result['identical'] = False
                    result['details']['key_mismatch'] = True
                    return result
                
                # Compare each dataset
                result['identical'] = True
                for key in keys1:
                    data1 = f1[key][:]
                    data2 = f2[key][:]
                    
                    # Store shapes for diagnostics
                    result['details'][f'{key}_shape1'] = data1.shape
                    result['details'][f'{key}_shape2'] = data2.shape
                    
                    # Check if shapes match first
                    if data1.shape != data2.shape:
                        result['identical'] = False
                        result['details'][f'{key}_identical'] = False
                        result['details'][f'{key}_shape_mismatch'] = True
                        self.logger.warning(f"QC_COMP: Shape mismatch in {filename}[{key}]: {data1.shape} vs {data2.shape}")
                        continue
                    
                    # Compare data if shapes match
                    try:
                        if not np.array_equal(data1, data2):
                            result['identical'] = False
                            result['details'][f'{key}_identical'] = False
                            
                            # Calculate difference metrics for numeric data
                            if np.issubdtype(data1.dtype, np.number) and np.issubdtype(data2.dtype, np.number):
                                diff = np.abs(data1 - data2)
                                result['details'][f'{key}_max_diff'] = float(np.max(diff))
                                result['details'][f'{key}_mean_diff'] = float(np.mean(diff))
                        else:
                            result['details'][f'{key}_identical'] = True
                    except Exception as e:
                        result['identical'] = False
                        result['details'][f'{key}_identical'] = False
                        result['details'][f'{key}_comparison_error'] = str(e)
                        self.logger.warning(f"QC_COMP: Comparison error in {filename}[{key}]: {e}")
        
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"QC_COMP: Error comparing {filename}: {e}")
        
        return result
    
    def compare_qc_outputs(self, original_path: str, pipeline_path: str) -> Dict[str, Any]:
        """
        Compare all QC outputs between original and pipeline processing.
        
        Args:
            original_path: Path to original QC output directory
            pipeline_path: Path to pipeline QC output directory
            
        Returns:
            Dictionary with comprehensive comparison results
        """
        self.logger.info(f"QC_COMP: Comparing QC outputs")
        self.logger.info(f"QC_COMP: Original: {original_path}")
        self.logger.info(f"QC_COMP: Pipeline: {pipeline_path}")
        
        results = {
            'comparison_summary': {
                'total_files': 0,
                'identical_files': 0,
                'different_files': 0,
                'missing_files': 0,
                'error_files': 0
            },
            'file_results': {}
        }
        
        # Files to compare
        files_to_compare = [
            ('qc_results/fluo.npy', 'npy'),
            ('qc_results/neuropil.npy', 'npy'),
            ('qc_results/stat.npy', 'npy'),
            ('qc_results/masks.npy', 'npy'),
            ('qc_results/qc_stats.npy', 'npy'),  # QC statistics
            ('ops.npy', 'npy'),
            ('move_offset.h5', 'h5'),
            ('masks.h5', 'h5'),  # Labeling results
            ('dff.h5', 'h5')  # dF/F traces
        ]
        
        for filename, file_type in files_to_compare:
            original_file = os.path.join(original_path, filename)
            pipeline_file = os.path.join(pipeline_path, filename)
            
            if file_type == 'npy':
                result = self.compare_npy_files(original_file, pipeline_file, filename)
            elif file_type == 'h5':
                result = self.compare_h5_files(original_file, pipeline_file, filename)
            
            results['file_results'][filename] = result
            results['comparison_summary']['total_files'] += 1
            
            # Update summary counts
            if result['error']:
                results['comparison_summary']['error_files'] += 1
                self.logger.error(f"QC_COMP: ❌ {filename}: {result['error']}")
            elif not result['both_exist']:
                results['comparison_summary']['missing_files'] += 1
                self.logger.warning(f"QC_COMP: ⚠️  {filename}: Missing file(s)")
            elif result['identical']:
                results['comparison_summary']['identical_files'] += 1
                self.logger.info(f"QC_COMP: ✅ {filename}: Files are identical")
            else:
                results['comparison_summary']['different_files'] += 1
                self.logger.warning(f"QC_COMP: ❌ {filename}: Files differ")
                if 'within_tolerance' in result['details'] and result['details']['within_tolerance']:
                    self.logger.info(f"QC_COMP:    └─ But differences are within tolerance")
        
        # Log summary
        summary = results['comparison_summary']
        self.logger.info(f"QC_COMP: === Comparison Summary ===")
        self.logger.info(f"QC_COMP: Total files: {summary['total_files']}")
        self.logger.info(f"QC_COMP: Identical: {summary['identical_files']}")
        self.logger.info(f"QC_COMP: Different: {summary['different_files']}")
        self.logger.info(f"QC_COMP: Missing: {summary['missing_files']}")
        self.logger.info(f"QC_COMP: Errors: {summary['error_files']}")
        
        return results
    
    def _compare_dicts(self, dict1: dict, dict2: dict) -> bool:
        """
        Compare two dictionaries, handling numpy arrays within them.
        
        Args:
            dict1: First dictionary
            dict2: Second dictionary
            
        Returns:
            True if dictionaries are equivalent, False otherwise
        """
        try:
            if set(dict1.keys()) != set(dict2.keys()):
                return False
            
            for key in dict1.keys():
                val1 = dict1[key]
                val2 = dict2[key]
                
                if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                    try:
                        if not np.array_equal(val1, val2):
                            return False
                    except (ValueError, TypeError):
                        # Handle cases where arrays have different shapes or types
                        if val1.shape != val2.shape:
                            return False
                        try:
                            # Try element-wise comparison
                            if not np.allclose(val1, val2, equal_nan=True):
                                return False
                        except (ValueError, TypeError):
                            # Final fallback to string comparison
                            if str(val1) != str(val2):
                                return False
                elif type(val1) != type(val2):
                    return False
                elif isinstance(val1, (int, float, str, bool)):
                    if val1 != val2:
                        return False
                elif hasattr(val1, '__array__') and hasattr(val2, '__array__'):
                    # Handle array-like objects
                    try:
                        if not np.array_equal(np.asarray(val1), np.asarray(val2)):
                            return False
                    except (ValueError, TypeError):
                        if str(val1) != str(val2):
                            return False
                else:
                    # For other types, try direct comparison first
                    try:
                        if val1 != val2:
                            return False
                    except (ValueError, TypeError):
                        # Fallback to string comparison
                        if str(val1) != str(val2):
                            return False
            
            return True
            
        except Exception as e:
            # If all else fails, return False and log the error
            if hasattr(self, 'logger'):
                self.logger.debug(f"QC_COMP: Dict comparison error: {e}")
            return False

    def _compare_object_arrays(self, arr1: np.ndarray, arr2: np.ndarray) -> bool:
        """
        Compare object arrays (like stat arrays) safely.
        
        Args:
            arr1: First object array
            arr2: Second object array
            
        Returns:
            True if arrays are identical, False otherwise
        """
        if arr1.shape != arr2.shape:
            return False
        
        if len(arr1) != len(arr2):
            return False
        
        for i in range(len(arr1)):
            try:
                item1 = arr1[i]
                item2 = arr2[i]
                
                if item1 is None and item2 is None:
                    continue
                elif item1 is None or item2 is None:
                    return False
                elif isinstance(item1, dict) and isinstance(item2, dict):
                    if not self._compare_dicts(item1, item2):
                        return False
                else:
                    # For non-dict objects, try safe comparison
                    try:
                        if hasattr(item1, '__array__') and hasattr(item2, '__array__'):
                            if not np.array_equal(np.asarray(item1), np.asarray(item2)):
                                return False
                        else:
                            if item1 != item2:
                                return False
                    except (ValueError, TypeError):
                        # Fallback to string comparison
                        if str(item1) != str(item2):
                            return False
            except Exception:
                # If any comparison fails, assume they're different
                return False
        
        return True

    def _get_dict_differences(self, dict1: dict, dict2: dict) -> Dict[str, Any]:
        """
        Get detailed information about differences between two dictionaries.
        
        Args:
            dict1: First dictionary
            dict2: Second dictionary
            
        Returns:
            Dictionary with difference details
        """
        differences = {
            'keys_only_in_dict1': [],
            'keys_only_in_dict2': [],
            'different_values': {},
            'different_types': {},
            'array_differences': {}
        }
        
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())
        
        differences['keys_only_in_dict1'] = list(keys1 - keys2)
        differences['keys_only_in_dict2'] = list(keys2 - keys1)
        
        common_keys = keys1 & keys2
        
        for key in common_keys:
            val1 = dict1[key]
            val2 = dict2[key]
            
            if type(val1) != type(val2):
                differences['different_types'][key] = {
                    'type1': str(type(val1)),
                    'type2': str(type(val2))
                }
            elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                try:
                    if not np.array_equal(val1, val2):
                        differences['array_differences'][key] = {
                            'shape1': val1.shape,
                            'shape2': val2.shape,
                            'dtype1': str(val1.dtype),
                            'dtype2': str(val2.dtype)
                        }
                        if val1.shape == val2.shape and np.issubdtype(val1.dtype, np.number):
                            diff = np.abs(val1 - val2)
                            differences['array_differences'][key]['max_diff'] = float(np.max(diff))
                            differences['array_differences'][key]['mean_diff'] = float(np.mean(diff))
                except (ValueError, TypeError):
                    differences['different_values'][key] = {
                        'val1_str': str(val1)[:100],  # Truncate for readability
                        'val2_str': str(val2)[:100]
                    }
            elif isinstance(val1, (int, float, str, bool)):
                if val1 != val2:
                    differences['different_values'][key] = {
                        'val1': val1,
                        'val2': val2
                    }
            else:
                try:
                    if val1 != val2:
                        differences['different_values'][key] = {
                            'val1_str': str(val1)[:100],
                            'val2_str': str(val2)[:100]
                        }
                except (ValueError, TypeError):
                    differences['different_values'][key] = {
                        'val1_str': str(val1)[:100],
                        'val2_str': str(val2)[:100]
                    }
        
        return differences

    def print_detailed_comparison(self, results: Dict[str, Any]):
        """
        Print detailed comparison results to console.
        
        Args:
            results: Results dictionary from compare_qc_outputs
        """
        print("\n" + "="*60)
        print("QC OUTPUT COMPARISON REPORT")
        print("="*60)
        
        for filename, result in results['file_results'].items():
            print(f"\n📁 {filename}")
            print("-" * 50)
            
            if result['error']:
                print(f"❌ ERROR: {result['error']}")
            elif not result['both_exist']:
                print("⚠️  MISSING: One or both files missing")
            elif result['identical']:
                print("✅ IDENTICAL: Files match perfectly")
            else:
                print("❌ DIFFERENT: Files do not match")
                
                details = result['details']
                if 'shape1' in details:
                    print(f"   Original shape: {details['shape1']}")
                    print(f"   Pipeline shape: {details['shape2']}")
                if 'dtype1' in details:
                    print(f"   Original dtype: {details['dtype1']}")
                    print(f"   Pipeline dtype: {details['dtype2']}")
                if 'max_diff' in details:
                    print(f"   Max difference: {details['max_diff']}")
                    print(f"   Mean difference: {details['mean_diff']}")
                    print(f"   Within tolerance: {details.get('within_tolerance', 'N/A')}")
                
                # Print H5 dataset details
                if filename.endswith('.h5'):
                    h5_keys = [k for k in details.keys() if k.endswith('_shape1')]
                    if h5_keys:
                        print("   H5 Dataset comparison:")
                        for key_info in h5_keys:
                            dataset_name = key_info.replace('_shape1', '')
                            shape1 = details.get(f'{dataset_name}_shape1', 'unknown')
                            shape2 = details.get(f'{dataset_name}_shape2', 'unknown')
                            identical = details.get(f'{dataset_name}_identical', False)
                            shape_mismatch = details.get(f'{dataset_name}_shape_mismatch', False)
                            
                            status = "✅" if identical else "❌"
                            print(f"     {status} {dataset_name}: {shape1} vs {shape2}")
                            
                            if shape_mismatch:
                                print(f"         └─ SHAPE MISMATCH")
                            elif not identical:
                                max_diff = details.get(f'{dataset_name}_max_diff')
                                if max_diff is not None:
                                    print(f"         └─ Max diff: {max_diff}")
                
                # Print detailed dictionary differences for ops.npy
                if 'dict_comparison' in details:
                    dict_diff = details['dict_comparison']
                    print("   Dictionary differences:")
                    
                    if dict_diff['keys_only_in_dict1']:
                        print(f"     Keys only in original: {dict_diff['keys_only_in_dict1']}")
                    if dict_diff['keys_only_in_dict2']:
                        print(f"     Keys only in pipeline: {dict_diff['keys_only_in_dict2']}")
                    
                    if dict_diff['different_types']:
                        print("     Different types:")
                        for key, types in dict_diff['different_types'].items():
                            print(f"       {key}: {types['type1']} vs {types['type2']}")
                    
                    if dict_diff['different_values']:
                        print("     Different values:")
                        for key, vals in list(dict_diff['different_values'].items())[:5]:  # Show first 5
                            print(f"       {key}: {vals.get('val1', vals.get('val1_str', 'N/A'))} vs {vals.get('val2', vals.get('val2_str', 'N/A'))}")
                        if len(dict_diff['different_values']) > 5:
                            print(f"       ... and {len(dict_diff['different_values']) - 5} more")
                    
                    if dict_diff['array_differences']:
                        print("     Array differences:")
                        for key, arr_diff in dict_diff['array_differences'].items():
                            print(f"       {key}: shape {arr_diff['shape1']} vs {arr_diff['shape2']}")
                            if 'max_diff' in arr_diff:
                                print(f"         Max diff: {arr_diff['max_diff']}")
        
        print("\n" + "="*60)
        summary = results['comparison_summary']
        print(f"SUMMARY: {summary['identical_files']}/{summary['total_files']} files identical")
        print("="*60)


def run_qc_comparison(original_path: str, pipeline_path: str, logger=None):
    """
    Convenience function to run QC comparison.
    
    Args:
        original_path: Path to original QC output directory
        pipeline_path: Path to pipeline QC output directory
        logger: Optional logger instance
        
    Returns:
        Comparison results dictionary
    """
    comparator = QCOutputComparator(logger)
    results = comparator.compare_qc_outputs(original_path, pipeline_path)
    comparator.print_detailed_comparison(results)
    return results


if __name__ == "__main__":
    # Example usage
    original_path = "D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/check_diff"
    pipeline_path = "D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494"
    
    results = run_qc_comparison(original_path, pipeline_path)
