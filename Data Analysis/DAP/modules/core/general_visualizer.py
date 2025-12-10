import os
import logging
import importlib
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from datetime import datetime

import modules.utils as utils

class GeneralVisualizer:
    """
    General visualizer that handles common visualization tasks and loads experiment-specific visualizers.
    Acts as both pipeline component and visualization infrastructure provider.
    """
    
    def __init__(self, config_manager, subject_list, analysis_results, run_id=None, logger=None):
        """
        Initialize the GeneralVisualizer with ConfigManager, subject list, and analysis results.
        """
        self.config_manager = config_manager
        self.subject_list = subject_list
        self.analysis_results = analysis_results
        self.run_id = run_id or datetime.now().strftime('%Y%m%d_%H%M%S')  # Fallback if not provided
        self.logger = logger
        
        self.logger.info("VZ: Initializing GeneralVisualizer...")
        
        # Get config
        self.config = config_manager.config
        
        # Setup visualization infrastructure
        self._setup_visualization_infrastructure()
        
        # Initialize experiment-specific visualizer if specified
        self.experiment_visualizer = self._load_experiment_visualizer()
        
        self.logger.info("VZ: GeneralVisualizer initialized successfully")

    def _setup_visualization_infrastructure(self):
        """Setup common visualization infrastructure."""
        # Setup output directories
        self._setup_output_directories()
        
        # Setup matplotlib
        self._setup_interactive_backend()
        self._setup_plot_style()
        
        # Figure tracking
        self.figure_metadata = []

    def _setup_output_directories(self):
        """Create organized output directory structure for figures that aligns with reports."""
        # Base output directory
        base_output = "output"
        
        # Main directories
        self.output_dirs = {
            'base': base_output,
            'figures': os.path.join(base_output, 'figures'),
            'reports': os.path.join(base_output, 'reports'),
            'subject_reports': os.path.join(base_output, 'reports', 'subjects'),
            'group_reports': os.path.join(base_output, 'reports', 'group'),
        }
        
        # Create main directories
        for dir_path in self.output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Create subject report directories with figure subdirectories
        for subject_id in self.subject_list:
            subject_report_dir = os.path.join(self.output_dirs['subject_reports'], subject_id)
            subject_figures_dir = os.path.join(subject_report_dir, 'figures')
            
            # Create subdirectories for different figure types
            figure_subdirs = [
                subject_figures_dir,  # Main figures folder
                os.path.join(subject_figures_dir, 'cross_session_analysis'),
                os.path.join(subject_figures_dir, 'session_overviews')
            ]
            
            for subdir in figure_subdirs:
                os.makedirs(subdir, exist_ok=True)
            
            self.logger.info(f"VIS: Created figure directories for subject {subject_id}")
        
        # Create group figures directory
        group_figures_dir = os.path.join(self.output_dirs['group_reports'], 'figures')
        os.makedirs(group_figures_dir, exist_ok=True)

    def _setup_interactive_backend(self):
        """Setup matplotlib backend - respect existing backend if in Spyder mode."""
        
        # Check visualization config settings
        viz_config = self.config.get('visualization', {})
        show_plots_config = viz_config.get('show_plots', True)
        spyder_mode = viz_config.get('spyder_mode', False)
        
        if spyder_mode:
            # In Spyder mode, don't change the backend - use what's already set
            current_backend = matplotlib.get_backend()
            self.interactive_mode = show_plots_config
            self.logger.info(f"VZ: Spyder mode - using existing backend: {current_backend}")
            
            if self.interactive_mode:
                plt.ioff()  # Spyder manages display, use non-interactive mode
                self.logger.info("VZ: Spyder mode - using non-interactive matplotlib")
            else:
                plt.ioff()
                self.logger.info("VZ: Spyder mode - plots disabled")
                
        elif not show_plots_config:
            # Headless mode
            matplotlib.use('Agg')
            self.interactive_mode = False
            plt.ioff()
            self.logger.info("VZ: Headless plotting enabled (show_plots disabled)")
        else:
            # Non-Spyder interactive mode
            try:
                matplotlib.use('TkAgg')
                self.interactive_mode = True
                plt.ion()
                self.logger.info("VZ: Interactive plotting enabled (development mode)")
            except Exception:
                matplotlib.use('Agg')
                self.interactive_mode = False
                plt.ioff()
                self.logger.info("VZ: Headless plotting enabled (TkAgg unavailable)")
        
        # Remove this conflicting line that was overriding interactive mode
        # plt.ioff()

    def _setup_plot_style(self):
        """Setup standardized plot styling."""
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'Arial',
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'lines.linewidth': 2,
            'axes.linewidth': 1.5,
            'figure.dpi': 150
        })
    
    def save_figure(self, fig, filename: str, subject_id: str = None, subfolder: str = None, 
                   description: str = ""):
        """Save figure to appropriate report directory structure."""
        try:
            # Determine output directory based on subject and subfolder
            if subject_id:
                if subfolder:
                    output_dir = os.path.join(self.output_dirs['subject_reports'], subject_id, 'figures', subfolder)
                else:
                    output_dir = os.path.join(self.output_dirs['subject_reports'], subject_id, 'figures')
            else:
                # Group-level figure
                output_dir = os.path.join(self.output_dirs['group_reports'], 'figures')
                if subfolder:
                    output_dir = os.path.join(output_dir, subfolder)
                    os.makedirs(output_dir, exist_ok=True)
            
            # Save figure
            figure_path = os.path.join(output_dir, filename)
            fig.savefig(figure_path, bbox_inches='tight', dpi=300)
            
            # Track metadata
            figure_info = {
                'path': figure_path,
                'filename': filename,
                'type': 'figure',  # Update type to 'figure'
                'subject': subject_id,
                'subfolder': subfolder,
                'description': description,
                'size': fig.get_size_inches().tolist()
            }
            self.figure_metadata.append(figure_info)
            
            self.logger.info(f"VZ: Figure saved: {figure_info}")
            
            return figure_path
        
        except Exception as e:
            self.logger.error(f"VZ: Failed to save figure: {e}")
            raise

    def show_figure(self, fig, show_plot=None):
        """Show figure if in interactive mode, with optional override for individual plots."""
        # Determine if we should show: use override if provided, otherwise use default behavior
        should_show = show_plot if show_plot is not None else self.interactive_mode
        
        if self.interactive_mode and should_show:
            # For Spyder, we need to actually show the figure to make it appear in the plot pane
            plt.figure(fig.number)  # Make this the current figure
            plt.show()  # This should now work with interactive mode to show in Spyder's plot pane
            print('')
        else:
            self.logger.debug(f"VZ: Figure display disabled (interactive_mode={self.interactive_mode}, show_plot={show_plot})")
    
    def _position_plot_window(self, fig):
        """Position plot window for convenient viewing - disabled for non-blocking display."""
        # Commenting out window positioning since we're using non-blocking display
        # This prevents popup windows in Spyder while still allowing plots to appear in plot pane
        pass
        # try:
        #     manager = fig.canvas.manager
        #     manager.window.wm_geometry("1024x768+0+0")
        #     fig.canvas.draw()
        #     fig.canvas.flush_events()
        # except Exception as e:
        #     self.logger.warning(f"VZ: Could not position plot window: {e}")

    def _load_experiment_visualizer(self):
        """Load experiment-specific visualizer if configured."""
        try:
            # Get experiment config
            experiment_config = self.config.get('experiment_configs', {}).get(self.config_manager.experiment_name, {})
            visualizer_module_name = experiment_config.get('visualizer_module')
            
            if visualizer_module_name is None:
                self.logger.info("VZ: No experiment-specific visualizer configured, using general visualization only")
                return None
            
            # Get visualizer info from available list
            available_visualizers = self.config.get('available_visualizers', {})
            visualizer_info = available_visualizers.get(visualizer_module_name)
            
            if not visualizer_info:
                self.logger.error(f"VZ: Visualizer module '{visualizer_module_name}' not found in available_visualizers")
                return None
            
            class_name = visualizer_info.get('class')
            if not class_name:
                self.logger.error(f"VZ: No class specified for visualizer module '{visualizer_module_name}'")
                return None
            
            self.logger.info(f"VZ: Loading experiment-specific visualizer: {class_name} from {visualizer_module_name}")
            
            # Direct import of the specific module and class
            module = importlib.import_module(f"modules.experiments.{self.config_manager.experiment_name}.{visualizer_module_name}")
            visualizer_class = getattr(module, class_name)
            
            # Initialize the experiment-specific visualizer, passing infrastructure
            experiment_visualizer = visualizer_class(self.config_manager, self.subject_list, self.analysis_results, self, logger=self.logger)
            self.logger.info(f"VZ: Successfully loaded experiment-specific visualizer: {class_name}")
            
            return experiment_visualizer
            
        except Exception as e:
            self.logger.error(f"VZ: Failed to load experiment-specific visualizer: {e}")
            self.logger.info("VZ: Falling back to general visualization only")
            return None

    def generate_visualizations(self) -> Dict[str, Any]:
        """
        Generate visualizations using experiment-specific visualizer if available.
        
        Returns:
            Visualization results dictionary
        """
        self.logger.info("VZ: Starting visualization generation...")
        
        if self.experiment_visualizer is not None:
            # Use experiment-specific visualization
            results = self.experiment_visualizer.generate_visualizations()
            
            # Add infrastructure metadata
            results.update({
                'figures_generated': len(self.figure_metadata),
                'figure_metadata': self.figure_metadata,
                'output_directories': self.output_dirs
            })
            
            return results
        else:
            # Use general visualization only
            self.logger.info("VZ: Applying general visualization...")
            return self._general_visualization()

    def _general_visualization(self) -> Dict[str, Any]:
        """
        Perform basic general visualization when no experiment-specific visualizer is available.
        """
        self.logger.info("VZ: Performing general visualization...")
        
        # Basic visualization - just return metadata
        visualization_results = {
            'visualization_type': 'general',
            'experiment_config': self.config_manager.experiment_name,
            'subjects_visualized': len(self.subject_list),
            'figures_generated': 0,
            'summary': 'No experiment-specific visualizer configured'
        }
        
        self.logger.info("VZ: General visualization completed")
        return visualization_results


