import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
import warnings

class MeasurementAnalyzer:
    def __init__(self):
        self.data = pd.DataFrame()
        
    def add_measurement(self, 
                       name: str,
                       values: np.ndarray,
                       **flags) -> None:
        """
        Add a measurement with associated flags
        
        Args:
            name: measurement name
            values: measurement values
            **flags: arbitrary flag columns (e.g., valid=True, outlier=False, etc.)
        """
        if len(self.data) == 0:
            # First measurement - initialize DataFrame
            self.data = pd.DataFrame({name: values})
        else:
            # Check length compatibility
            if len(values) != len(self.data):
                raise ValueError(f"Length mismatch: {len(values)} vs {len(self.data)}")
            self.data[name] = values
            
        # Add flag columns
        for flag_name, flag_values in flags.items():
            col_name = f"{name}_{flag_name}"
            if np.isscalar(flag_values):
                # Broadcast scalar to all rows
                self.data[col_name] = flag_values
            else:
                if len(flag_values) != len(values):
                    raise ValueError(f"Flag length mismatch for {flag_name}")
                self.data[col_name] = flag_values
    
    def set_flags(self, 
                  measurement: str, 
                  **flags) -> None:
        """Set flags for existing measurement"""
        if measurement not in self.data.columns:
            raise ValueError(f"Measurement {measurement} not found")
            
        for flag_name, flag_values in flags.items():
            col_name = f"{measurement}_{flag_name}"
            if np.isscalar(flag_values):
                self.data[col_name] = flag_values
            else:
                if len(flag_values) != len(self.data):
                    raise ValueError(f"Flag length mismatch for {flag_name}")
                self.data[col_name] = flag_values
    
    def get_mask(self, **conditions) -> pd.Series:
        """
        Get boolean mask based on conditions
        
        Args:
            **conditions: conditions like measurement_flag=value
            
        Returns:
            Boolean mask
        """
        mask = pd.Series(True, index=self.data.index)
        
        for condition, value in conditions.items():
            if condition not in self.data.columns:
                raise ValueError(f"Column {condition} not found")
            mask &= (self.data[condition] == value)
            
        return mask
    
    def get_filtered_data(self, 
                         measurements: Optional[List[str]] = None,
                         **conditions) -> pd.DataFrame:
        """Get filtered data based on conditions"""
        mask = self.get_mask(**conditions)
        
        if measurements is None:
            # Return all measurement columns (not flag columns)
            measurement_cols = [col for col in self.data.columns 
                              if not any('_' in col and col.split('_', 1)[1] in 
                                       ['valid', 'outlier', 'quality', 'flag'] 
                                       for _ in [None])]
            # Better approach: track measurement columns
            measurement_cols = [col for col in self.data.columns 
                              if not '_' in col or 
                              not any(col.endswith(f'_{flag}') 
                                    for flag in self._get_flag_suffixes())]
            return self.data.loc[mask, measurement_cols]
        else:
            return self.data.loc[mask, measurements]
    
    def _get_flag_suffixes(self) -> set:
        """Get all flag suffixes used in the dataset"""
        flag_suffixes = set()
        for col in self.data.columns:
            if '_' in col:
                parts = col.split('_')
                if len(parts) >= 2:
                    flag_suffixes.add(parts[-1])
        return flag_suffixes
    
    def get_measurement_columns(self) -> List[str]:
        """Get list of measurement columns (excluding flags)"""
        flag_cols = set()
        measurement_cols = []
        
        # First pass: identify all flag columns
        for col in self.data.columns:
            if '_' in col:
                base_name = col.rsplit('_', 1)[0]
                if base_name in self.data.columns:
                    flag_cols.add(col)
        
        # Second pass: get measurement columns
        for col in self.data.columns:
            if col not in flag_cols:
                measurement_cols.append(col)
                
        return measurement_cols
    
    def summary(self, measurement: str) -> pd.DataFrame:
        """Get summary statistics grouped by flags"""
        if measurement not in self.data.columns:
            raise ValueError(f"Measurement {measurement} not found")
        
        # Find flag columns for this measurement
        flag_cols = [col for col in self.data.columns 
                    if col.startswith(f"{measurement}_")]
        
        if not flag_cols:
            # No flags, just basic stats
            return self.data[measurement].describe().to_frame().T
        
        # Group by flag combinations and compute stats
        summary_data = []
        for flag_col in flag_cols:
            for flag_value in self.data[flag_col].unique():
                mask = self.data[flag_col] == flag_value
                stats = self.data.loc[mask, measurement].describe()
                stats['flag'] = f"{flag_col}={flag_value}"
                stats['count_total'] = len(self.data)
                summary_data.append(stats)
        
        return pd.DataFrame(summary_data).set_index('flag')
    
    def plot(self, 
             x_col: str, 
             y_col: str, 
             color_by: Optional[str] = None,
             filter_conditions: Optional[Dict] = None,
             **plot_kwargs) -> plt.Figure:
        """Plot measurements with optional coloring by flags"""
        
        # Apply filters
        if filter_conditions:
            mask = self.get_mask(**filter_conditions)
            plot_data = self.data[mask]
        else:
            plot_data = self.data
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if color_by and color_by in plot_data.columns:
            # Color by flag values
            for value in plot_data[color_by].unique():
                mask = plot_data[color_by] == value
                subset = plot_data[mask]
                ax.scatter(subset[x_col], subset[y_col], 
                          label=f"{color_by}={value}", 
                          alpha=0.7, **plot_kwargs)
            ax.legend()
        else:
            ax.scatter(plot_data[x_col], plot_data[y_col], 
                      alpha=0.7, **plot_kwargs)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")
        
        return fig

# Example usage
if __name__ == "__main__":
    # Create analyzer
    analyzer = MeasurementAnalyzer()
    
    # Generate sample data
    n_points = 1000
    time = np.linspace(0, 10, n_points)
    
    # Temperature measurement with some issues
    temp = 20 + 5 * np.sin(time) + np.random.normal(0, 0.5, n_points)
    temp_outliers = np.random.random(n_points) < 0.05
    temp[temp_outliers] += np.random.normal(0, 10, np.sum(temp_outliers))
    temp_valid = ~temp_outliers & (temp > 10) & (temp < 35)
    temp_quality = np.random.choice(['good', 'fair', 'poor'], n_points, p=[0.7, 0.2, 0.1])
    
    # Pressure measurement
    pressure = 1013 + 10 * np.cos(time * 0.5) + np.random.normal(0, 1, n_points)
    pressure_valid = (pressure > 900) & (pressure < 1100)
    pressure_sensor_ok = np.random.random(n_points) > 0.03
    
    # Add measurements with flags
    analyzer.add_measurement('time', time)
    analyzer.add_measurement('temperature', temp, 
                           valid=temp_valid, 
                           outlier=temp_outliers,
                           quality=temp_quality)
    analyzer.add_measurement('pressure', pressure,
                           valid=pressure_valid,
                           sensor_ok=pressure_sensor_ok)
    
    print("Dataset shape:", analyzer.data.shape)
    print("\nColumns:", list(analyzer.data.columns))
    print("\nMeasurement columns:", analyzer.get_measurement_columns())
    
    # Filter examples
    print("\n=== Valid temperature data ===")
    valid_temp = analyzer.get_filtered_data(temperature_valid=True)
    print(f"Valid temperature points: {len(valid_temp)}")
    
    print("\n=== High quality, valid data ===")
    good_data = analyzer.get_filtered_data(
        temperature_valid=True,
        temperature_quality='good',
        pressure_valid=True,
        pressure_sensor_ok=True
    )
    print(f"Good quality points: {len(good_data)}")
    
    # Summary statistics
    print("\n=== Temperature summary by quality ===")
    temp_summary = analyzer.summary('temperature')
    print(temp_summary[['count', 'mean', 'std']])
    
    # Plotting
    fig1 = analyzer.plot('time', 'temperature', color_by='temperature_valid')
    plt.title('Temperature vs Time (colored by validity)')
    
    fig2 = analyzer.plot('time', 'temperature', 
                        color_by='temperature_quality',
                        filter_conditions={'temperature_valid': True})
    plt.title('Valid Temperature vs Time (colored by quality)')
    
    plt.show()
    
    # Easy to add new flags
    print("\n=== Adding new flags ===")
    analyzer.set_flags('temperature', 
                      high_temp=analyzer.data['temperature'] > 25,
                      morning=analyzer.data['time'] < 5)
    
    morning_hot = analyzer.get_filtered_data(
        temperature_morning=True,
        temperature_high_temp=True,
        temperature_valid=True
    )
    print(f"Hot morning points: {len(morning_hot)}")