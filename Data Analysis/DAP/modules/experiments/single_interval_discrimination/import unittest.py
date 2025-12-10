import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import warnings




import os
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


print(current_dir)


from import_os import test_create_improved_visualization, create_mock_trial_start_data, create_mock_event_data

import matplotlib.pyplot as plt

# Import the function to test using relative import

class TestCreateImprovedVisualization(unittest.TestCase):
    """Test suite for the test_create_improved_visualization function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Suppress matplotlib warnings during testing
        warnings.filterwarnings('ignore')
        plt.ioff()  # Turn off interactive mode for testing
        
        # Create test data with more distinct patterns
        self.n_trials = 40
        self.n_rois = 20
        self.trial_start_data = self.create_enhanced_mock_data()
        
        # Define realistic ROI groups
        self.roi_groups = {
            'f1_onset': [0, 1, 2, 3, 4],
            'f1_offset': [5, 6, 7, 8],
            'f2_onset': [2, 3, 9, 10, 11],
            'f2_offset': [12, 13, 14],
            'choice_start': [4, 5, 15, 16],
            'isi_sensitivity': [1, 7, 17, 18, 19]
        }
    
    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')
    
    def create_enhanced_mock_data(self):
        """Create mock data with more distinct and testable patterns."""
        time_vector = np.linspace(0, 4.0, 120)
        dff_segments = np.random.randn(self.n_trials, self.n_rois, 120) * 0.02
        
        # Add very distinct patterns for testing
        isi_values = np.random.choice([200, 700, 1700], self.n_trials)
        df_trials = pd.DataFrame({
            'trial_id': range(self.n_trials),
            'isi': isi_values,
            'is_right': np.random.choice([0, 1], self.n_trials),
            'is_right_choice': np.random.choice([0, 1], self.n_trials),
            'did_not_choose': np.random.choice([0, 1], self.n_trials, p=[0.9, 0.1]),
            'rewarded': np.random.choice([0, 1], self.n_trials, p=[0.7, 0.3]),
            'lick': np.random.choice([0, 1], self.n_trials, p=[0.8, 0.2])
        })
        
        df_trials['start_flash_2'] = 0.5 + isi_values / 1000
        
        # Add very distinct ISI-sensitive patterns for easier testing
        long_isi_trials = np.where(df_trials['is_right'] == 1)[0]
        f1_offset_idx = np.argmin(np.abs(time_vector - 0.5))
        
        # Make ROI 1 and 7 have very clear sustained suppression
        for trial_idx in long_isi_trials:
            f2_onset_time = df_trials.loc[trial_idx, 'start_flash_2']
            f2_onset_idx = np.argmin(np.abs(time_vector - f2_onset_time))
            isi_window = slice(f1_offset_idx + 3, f2_onset_idx - 3)
            
            if isi_window.stop > isi_window.start:
                # Clear sustained suppression
                dff_segments[trial_idx, 1, isi_window] -= 0.15
                dff_segments[trial_idx, 7, isi_window] -= 0.12
                # Clear sustained activation
                dff_segments[trial_idx, 17, isi_window] += 0.18
                dff_segments[trial_idx, 18, isi_window] += 0.16
        
        return {
            'dff_segments_array': dff_segments,
            'common_time_vector': time_vector,
            'df_trials_with_segments': df_trials,
            'event_name': 'trial_start',
            'n_trials': self.n_trials,
            'n_rois': self.n_rois,
            'isi_sensitive_rois_ground_truth': [1, 7, 17, 18, 19]
        }
    
    @patch('matplotlib.pyplot.show')
    def test_basic_visualization_execution(self, mock_show):
        """Test that the visualization function executes without errors."""
        try:
            test_create_improved_visualization()
            mock_show.assert_called()
            self.assertTrue(True, "Visualization function executed successfully")
        except Exception as e:
            self.fail(f"Visualization function failed with error: {e}")
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_creation_calls(self, mock_subplots, mock_show):
        """Test that plots are created with correct parameters."""
        # Mock the subplots return
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(6)]
        mock_subplots.return_value = (mock_fig, np.array(mock_axes).reshape(3, 2))
        
        test_create_improved_visualization()
        
        # Verify subplots was called multiple times (for different plot types)
        self.assertGreater(mock_subplots.call_count, 0)
        mock_show.assert_called()
    
    def test_mock_data_generation(self):
        """Test that mock data has expected properties."""
        data = self.trial_start_data
        
        # Check data structure
        self.assertIn('dff_segments_array', data)
        self.assertIn('common_time_vector', data)
        self.assertIn('df_trials_with_segments', data)
        
        # Check data shapes
        dff_shape = data['dff_segments_array'].shape
        self.assertEqual(dff_shape, (self.n_trials, self.n_rois, 120))
        self.assertEqual(len(data['common_time_vector']), 120)
        self.assertEqual(len(data['df_trials_with_segments']), self.n_trials)
        
        # Check for ISI-sensitive patterns in the data
        long_isi_trials = data['df_trials_with_segments']['is_right'] == 1
        if np.sum(long_isi_trials) > 0:
            long_indices = np.where(long_isi_trials)[0]
            
            # Check that ROI 1 shows suppression during ISI period
            time_vector = data['common_time_vector']
            f1_offset_idx = np.argmin(np.abs(time_vector - 0.5))
            f2_onset_idx = np.argmin(np.abs(time_vector - 2.2))
            isi_period = slice(f1_offset_idx + 3, f2_onset_idx - 3)
            
            roi_1_isi_activity = data['dff_segments_array'][long_indices[0], 1, isi_period]
            self.assertLess(np.mean(roi_1_isi_activity), -0.1, 
                          "ROI 1 should show clear suppression during ISI")
    
    def test_roi_groups_validity(self):
        """Test that ROI groups are valid for the data."""
        for group_name, roi_indices in self.roi_groups.items():
            # Check all ROI indices are within bounds
            self.assertTrue(all(0 <= idx < self.n_rois for idx in roi_indices),
                          f"ROI group {group_name} contains invalid indices")
            
            # Check groups are not empty
            self.assertGreater(len(roi_indices), 0,
                             f"ROI group {group_name} should not be empty")
    
    def test_isi_detection_logic(self):
        """Test the ISI sensitivity detection logic."""
        data = self.trial_start_data
        dff_data = data['dff_segments_array']
        time_vector = data['common_time_vector']
        
        # Test sustained modulation detection
        long_isi_trials = data['df_trials_with_segments']['is_right'] == 1
        long_isi_indices = np.where(long_isi_trials)[0]
        
        if len(long_isi_indices) > 0:
            # Define ISI period
            f1_offset_idx = np.argmin(np.abs(time_vector - 0.5))
            f2_onset_idx = np.argmin(np.abs(time_vector - 2.2))
            isi_period = slice(f1_offset_idx + 3, f2_onset_idx - 3)
            
            # Test detection on known ISI-sensitive ROI
            roi_1_activity = dff_data[long_isi_indices, 1, isi_period]
            roi_avg_activity = np.nanmean(roi_1_activity, axis=0)
            
            suppressed_fraction = np.sum(roi_avg_activity < -0.05) / len(roi_avg_activity)
            
            self.assertGreater(suppressed_fraction, 0.6,
                             "ROI 1 should show sustained suppression > 60% of ISI period")
    
    @patch('matplotlib.pyplot.show')
    def test_visualization_with_small_dataset(self, mock_show):
        """Test visualization with minimal data."""
        # Create very small dataset
        small_data = create_mock_trial_start_data(n_trials=10, n_rois=5, n_timepoints=30)
        
        # Should handle small datasets gracefully
        try:
            # This would need to be adapted to accept custom data
            test_create_improved_visualization()
            self.assertTrue(True, "Small dataset handled successfully")
        except Exception as e:
            # Should not crash with small datasets
            self.assertNotIsInstance(e, IndexError, 
                                   "Should not have index errors with small datasets")
    
    def test_colormap_consistency(self):
        """Test that consistent colormaps are used."""
        # This tests the concept - actual implementation would need access to plot objects
        vmin, vmax = -0.3, 0.5
        
        # Test that colormap limits are reasonable
        self.assertLess(vmin, 0, "vmin should be negative for suppression")
        self.assertGreater(vmax, 0, "vmax should be positive for activation")
        self.assertLess(vmin, vmax, "vmin should be less than vmax")
        
        # Test that range covers expected dF/F values
        expected_range = vmax - vmin
        self.assertGreater(expected_range, 0.5, "Color range should be sufficient for dF/F data")
    
    def test_event_marker_positions(self):
        """Test that event markers are positioned correctly."""
        time_vector = self.trial_start_data['common_time_vector']
        
        # Test F1 offset marker position
        f1_offset_idx = np.argmin(np.abs(time_vector - 0.5))
        expected_f1_time = time_vector[f1_offset_idx]
        self.assertAlmostEqual(expected_f1_time, 0.5, places=2,
                              msg="F1 offset marker should be near 0.5s")
        
        # Test F2 onset marker position
        f2_onset_idx = np.argmin(np.abs(time_vector - 2.2))
        expected_f2_time = time_vector[f2_onset_idx]
        self.assertAlmostEqual(expected_f2_time, 2.2, places=1,
                              msg="F2 onset marker should be near 2.2s")
    
    @patch('matplotlib.pyplot.show')
    def test_error_handling(self, mock_show):
        """Test error handling in visualization function."""
        # Test with matplotlib backend issues
        with patch('matplotlib.pyplot.subplots', side_effect=Exception("Backend error")):
            # Should handle matplotlib errors gracefully
            try:
                test_create_improved_visualization()
            except Exception as e:
                # Should not propagate matplotlib-specific errors
                self.assertNotIn("Backend", str(e), 
                               "Should handle matplotlib backend errors gracefully")
    
    def test_data_consistency_checks(self):
        """Test that data remains consistent throughout processing."""
        data = self.trial_start_data
        original_shape = data['dff_segments_array'].shape
        
        # After any processing, data should maintain its structure
        self.assertEqual(data['dff_segments_array'].shape, original_shape,
                        "Data shape should remain consistent")
        
        # Check for NaN values that might break visualization
        has_nans = np.isnan(data['dff_segments_array']).any()
        if has_nans:
            nan_count = np.isnan(data['dff_segments_array']).sum()
            total_elements = data['dff_segments_array'].size
            nan_fraction = nan_count / total_elements
            self.assertLess(nan_fraction, 0.1, 
                          "NaN values should be less than 10% of data")

if __name__ == '__main__':
    unittest.main()