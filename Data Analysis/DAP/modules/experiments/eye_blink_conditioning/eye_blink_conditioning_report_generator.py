import os
import yaml
from datetime import datetime

class EyeBlinkConditioningReportGenerator:
    def __init__(self, config_manager, subject_list, analysis_results, infrastructure, logger):
        self.config_manager = config_manager
        self.subject_list = subject_list
        self.analysis_results = analysis_results
        self.infrastructure = infrastructure
        self.logger = logger
        
        # Load report config
        with open('modules/reports/eye_blink_conditioning.yaml', 'r') as f:
            self.report_config = yaml.safe_load(f)

    def generate_reports(self):
        reports_generated = 0
        
        for subject_id in self.subject_list:
            # Get subject data
            if subject_id in self.analysis_results.get('subject_results', {}):
                subject_data = self.analysis_results['subject_results'][subject_id]
                
                # Build sections
                sections = []
                for section_config in self.report_config['report']['sections']:
                    if section_config['type'] == 'figure_list':
                        section = self._build_figure_list(section_config, subject_id)
                    elif section_config['type'] == 'session_figures':
                        section = self._build_session_figures(section_config, subject_id, subject_data)
                    sections.append(section)
                
                # Generate report
                context = {
                    'report_title': f"{self.report_config['report']['title']} - {subject_id}",
                    'sections': sections
                }
                
                self.infrastructure.save_html_report(
                    self.report_config['report']['template'], 
                    context, 
                    f'{subject_id}_report.html', 
                    subject_id
                )
                reports_generated += 1
        
        return {'reports_generated': reports_generated}

    def _build_figure_list(self, config, subject_id):
        figures = []
        for fig_config in config['figures']:
            pattern = fig_config['pattern'].format(subject_id=subject_id)
            subfolder = fig_config.get('subfolder', '')
            
            if subfolder:
                path = os.path.join('figures', subfolder, pattern)
                abs_path = os.path.join('output', 'reports', 'subjects', subject_id, 'figures', subfolder, pattern)
            else:
                path = os.path.join('figures', pattern)
                abs_path = os.path.join('output', 'reports', 'subjects', subject_id, 'figures', pattern)
            
            figures.append({
                'path': path,
                'exists': os.path.exists(abs_path),
                'row_type': fig_config.get('row_type', 'single')  # Add row_type
            })
        
        return {
            'title': config['title'],
            'type': config['type'],
            'figures': figures
        }

    def _build_session_figures(self, config, subject_id, subject_data):
        session_figures = {}
        
        # Get session dates from session names
        session_summaries = subject_data.get('session_summaries', {})
        session_dates = []
        for session_name in session_summaries.keys():
            import re
            date_match = re.search(r'(\d{8})', session_name)
            if date_match:
                session_dates.append(date_match.group(1))
        
        session_dates = sorted(list(set(session_dates)))
        
        for session_date in session_dates:
            figures = []
            for fig_config in config['figures']:
                pattern = fig_config['pattern'].format(subject_id=subject_id, session_date=session_date)
                subfolder = fig_config.get('subfolder', '')
                
                if subfolder:
                    path = os.path.join('figures', subfolder, pattern)
                    abs_path = os.path.join('output', 'reports', 'subjects', subject_id, 'figures', subfolder, pattern)
                else:
                    path = os.path.join('figures', pattern)
                    abs_path = os.path.join('output', 'reports', 'subjects', subject_id, 'figures', pattern)
                
                figures.append({
                    'path': path,
                    'exists': os.path.exists(abs_path),
                    'row_type': fig_config.get('row_type', 'single')  # Add row_type
                })
            
            session_figures[session_date] = figures
        
        return {
            'title': config['title'],
            'type': config['type'],
            'session_figures': session_figures
        }
