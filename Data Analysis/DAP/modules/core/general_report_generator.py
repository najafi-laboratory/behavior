import os
import importlib
from datetime import datetime

class GeneralReportGenerator:
    """
    Minimal general report generator that loads experiment-specific report generators.
    """
    
    def __init__(self, config_manager, subject_list, analysis_results, run_id=None, logger=None):
        self.config_manager = config_manager
        self.subject_list = subject_list
        self.analysis_results = analysis_results
        self.run_id = run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = logger
        
        # Setup output directories
        self._setup_output_directories()
        
        # Load experiment-specific report generator
        self.experiment_report_generator = self._load_experiment_report_generator()

    def _setup_output_directories(self):
        """Create basic output directory structure."""
        base_output = "output/reports"
        
        self.output_dirs = {
            'reports': base_output,
            'subject_reports': os.path.join(base_output, 'subjects'),
        }
        
        # Create directories
        for dir_path in self.output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Create subject directories with figures subdirectories
        for subject_id in self.subject_list:
            subject_dir = os.path.join(self.output_dirs['subject_reports'], subject_id)
            figures_dir = os.path.join(subject_dir, 'figures')
            
            os.makedirs(subject_dir, exist_ok=True)
            os.makedirs(figures_dir, exist_ok=True)
            os.makedirs(os.path.join(figures_dir, 'cross_session_analysis'), exist_ok=True)
            os.makedirs(os.path.join(figures_dir, 'session_overviews'), exist_ok=True)

    def _load_experiment_report_generator(self):
        """Load experiment-specific report generator."""
        try:
            experiment_name = self.config_manager.experiment_name
            module_name = f"{experiment_name}_report_generator"
            class_name = f"{experiment_name.title().replace('_', '')}ReportGenerator"
            
            module = importlib.import_module(f"modules.experiments.{self.config_manager.experiment_name}.{module_name}")
            report_generator_class = getattr(module, class_name)
            
            return report_generator_class(
                self.config_manager, self.subject_list, self.analysis_results, 
                self, logger=self.logger
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load experiment-specific report generator: {e}")
            return None

    def generate_reports(self):
        """Generate reports using experiment-specific generator."""
        if self.experiment_report_generator:
            return self.experiment_report_generator.generate_reports()
        else:
            self.logger.warning("No experiment-specific report generator available")
            return {'reports_generated': 0}

    def save_html_report(self, template_name, context, filename, subject_id):
        """Simple HTML report saving."""
        try:
            # Get template path
            experiment_name = self.config_manager.experiment_name
            template_path = f'modules/templates/{experiment_name}/{template_name}'
            
            # Load and render template
            from jinja2 import Environment, FileSystemLoader
            env = Environment(loader=FileSystemLoader(f'modules/templates/{experiment_name}'))
            template = env.get_template(template_name)
            html_content = template.render(**context)
            
            # Save HTML file
            if subject_id:
                output_path = os.path.join(self.output_dirs['subject_reports'], subject_id, filename)
            else:
                output_path = os.path.join(self.output_dirs['reports'], filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Saved HTML report: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to save HTML report: {e}")
            return None
