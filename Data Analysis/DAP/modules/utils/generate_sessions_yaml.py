"""
Session YAML Generator

Standalone script that scans subject session data folders and generates a YAML file
with sessions ordered from most recent to least recent for each subject.

Usage:
    python generate_sessions_yaml.py

The script will:
1. Scan the session data directory for subject folders
2. Extract session files from each subject folder
3. Parse session timestamps to determine chronological order
4. Generate a YAML file with sessions ordered most recent first
5. Save the output to the output directory
"""

import os
import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class SessionYAMLGenerator:
    """Generates YAML configuration with ordered session lists."""
    
    def __init__(self, session_data_path: str = None, output_dir: str = None):
        """
        Initialize the generator.
        
        Args:
            session_data_path: Path to session data directory. If None, uses default.
            output_dir: Path to output directory. If None, uses default.
        """
        # Default paths based on config structure
        self.session_data_path = session_data_path or "D://behavior//session_data"
        self.output_dir = output_dir or "output"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Session data path: {self.session_data_path}")
        print(f"Output directory: {self.output_dir}")
    
    def parse_session_timestamp(self, session_name: str) -> datetime:
        """
        Parse timestamp from session filename.
        
        Expected format: [PREFIX]_YYYYMMDD_HHMMSS
        
        Args:
            session_name: Session filename
            
        Returns:
            datetime object, or None if parsing fails
        """
        # Look for pattern: YYYYMMDD_HHMMSS at the end of filename
        timestamp_pattern = r'(\d{8})_(\d{6})$'
        match = re.search(timestamp_pattern, session_name)
        
        if match:
            date_str = match.group(1)  # YYYYMMDD
            time_str = match.group(2)  # HHMMSS
            
            try:
                # Parse into datetime
                dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                return dt
            except ValueError as e:
                print(f"Warning: Could not parse timestamp from '{session_name}': {e}")
                return None
        else:
            print(f"Warning: No timestamp pattern found in '{session_name}'")
            return None
    
    def get_subject_sessions(self, subject_folder: str) -> List[str]:
        """
        Get all session files for a subject folder.
        
        Args:
            subject_folder: Path to subject's session folder
            
        Returns:
            List of session filenames (without path)
        """
        if not os.path.exists(subject_folder):
            print(f"Warning: Subject folder not found: {subject_folder}")
            return []
        
        sessions = []
        try:
            # Get all items in the folder
            for item in os.listdir(subject_folder):
                item_path = os.path.join(subject_folder, item)
                
                # Check if it's a directory (session folder) or file
                if os.path.isdir(item_path):
                    # It's a session directory
                    sessions.append(item)
                elif os.path.isfile(item_path):
                    # It's a session file - extract name without extension
                    name_without_ext = os.path.splitext(item)[0]
                    sessions.append(name_without_ext)
            
            print(f"Found {len(sessions)} sessions in {subject_folder}")
            return sessions
            
        except PermissionError:
            print(f"Warning: Permission denied accessing {subject_folder}")
            return []
        except Exception as e:
            print(f"Warning: Error reading {subject_folder}: {e}")
            return []
    
    def order_sessions_by_timestamp(self, sessions: List[str]) -> List[str]:
        """
        Order sessions from most recent to least recent based on timestamps.
        
        Args:
            sessions: List of session names
            
        Returns:
            List of session names ordered by timestamp (newest first)
        """
        # Parse timestamps and create (datetime, session_name) tuples
        sessions_with_timestamps = []
        sessions_without_timestamps = []
        
        for session in sessions:
            timestamp = self.parse_session_timestamp(session)
            if timestamp:
                sessions_with_timestamps.append((timestamp, session))
            else:
                sessions_without_timestamps.append(session)
        
        # Sort by timestamp (newest first)
        sessions_with_timestamps.sort(key=lambda x: x[0], reverse=True)
        
        # Extract just the session names
        ordered_sessions = [session for _, session in sessions_with_timestamps]
        
        # Add sessions without timestamps at the end
        ordered_sessions.extend(sorted(sessions_without_timestamps))
        
        print(f"Ordered {len(ordered_sessions)} sessions by timestamp")
        if sessions_without_timestamps:
            print(f"Warning: {len(sessions_without_timestamps)} sessions without valid timestamps")
        
        return ordered_sessions
    
    def scan_subjects(self) -> Dict[str, List[str]]:
        """
        Scan the session data directory for all subjects and their sessions.
        
        Returns:
            Dictionary mapping subject_id -> ordered list of sessions
        """
        if not os.path.exists(self.session_data_path):
            print(f"Error: Session data path not found: {self.session_data_path}")
            return {}
        
        subjects_data = {}
        
        try:
            # Get all subject folders
            subject_folders = [f for f in os.listdir(self.session_data_path) 
                             if os.path.isdir(os.path.join(self.session_data_path, f))]
            
            print(f"Found {len(subject_folders)} subject folders")
            
            for subject_id in sorted(subject_folders):
                print(f"\nProcessing subject: {subject_id}")
                
                subject_folder_path = os.path.join(self.session_data_path, subject_id)
                
                # Get sessions for this subject
                sessions = self.get_subject_sessions(subject_folder_path)
                
                if sessions:
                    # Order sessions by timestamp
                    ordered_sessions = self.order_sessions_by_timestamp(sessions)
                    subjects_data[subject_id] = ordered_sessions
                    
                    print(f"  → {len(ordered_sessions)} sessions ordered")
                    if ordered_sessions:
                        print(f"  → Most recent: {ordered_sessions[0]}")
                        print(f"  → Oldest: {ordered_sessions[-1]}")
                else:
                    print(f"  → No sessions found")
            
            return subjects_data
            
        except Exception as e:
            print(f"Error scanning subjects: {e}")
            return {}
    
    def generate_yaml_config(self, subjects_data: Dict[str, List[str]]) -> Dict:
        """
        Generate YAML configuration structure.
        
        Args:
            subjects_data: Dictionary mapping subject_id -> session list
            
        Returns:
            Dictionary ready for YAML serialization
        """
        config = {
            'metadata': {
                'generated_by': 'generate_sessions_yaml.py',
                'generated_at': datetime.now().isoformat(),
                'source_path': self.session_data_path,
                'total_subjects': len(subjects_data),
                'total_sessions': sum(len(sessions) for sessions in subjects_data.values())
            },
            'subjects': {}
        }
        
        # Add each subject with their ordered sessions
        for subject_id, sessions in subjects_data.items():
            config['subjects'][subject_id] = {
                'sessions_to_process': sessions
            }
        
        return config
    
    def save_yaml_file(self, config: Dict, filename: str = None) -> str:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            filename: Output filename. If None, generates timestamped name.
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sessions_config_{timestamp}.yaml"
        
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(config, f, 
                         default_flow_style=False, 
                         sort_keys=False,
                         indent=2,
                         width=120)
            
            print(f"\nYAML configuration saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error saving YAML file: {e}")
            return None
    
    def print_summary(self, subjects_data: Dict[str, List[str]]):
        """Print summary of discovered sessions."""
        print(f"\n{'='*60}")
        print("SESSION DISCOVERY SUMMARY")
        print(f"{'='*60}")
        
        print(f"Session data path: {self.session_data_path}")
        print(f"Subjects found: {len(subjects_data)}")
        
        total_sessions = sum(len(sessions) for sessions in subjects_data.values())
        print(f"Total sessions: {total_sessions}")
        
        print(f"\nSubjects and session counts:")
        for subject_id, sessions in subjects_data.items():
            print(f"  {subject_id}: {len(sessions)} sessions")
            if sessions:
                # Show most recent session
                most_recent = sessions[0]
                timestamp = self.parse_session_timestamp(most_recent)
                if timestamp:
                    print(f"    → Most recent: {most_recent} ({timestamp.strftime('%Y-%m-%d %H:%M:%S')})")
                else:
                    print(f"    → Most recent: {most_recent}")
        
        print(f"{'='*60}")
    
    def run(self, output_filename: str = None) -> str:
        """
        Run the complete session discovery and YAML generation process.
        
        Args:
            output_filename: Optional custom output filename
            
        Returns:
            Path to generated YAML file
        """
        print("Starting session discovery...")
        
        # Scan for subjects and sessions
        subjects_data = self.scan_subjects()
        
        if not subjects_data:
            print("No subjects or sessions found. Exiting.")
            return None
        
        # Print summary
        self.print_summary(subjects_data)
        
        # Generate YAML configuration
        print("\nGenerating YAML configuration...")
        config = self.generate_yaml_config(subjects_data)
        
        # Save to file
        output_path = self.save_yaml_file(config, output_filename)
        
        if output_path:
            print(f"\n✓ Session YAML generation completed successfully!")
            print(f"  → Output file: {output_path}")
            print(f"  → {len(subjects_data)} subjects processed")
            print(f"  → {sum(len(s) for s in subjects_data.values())} total sessions")
        else:
            print("✗ Failed to save YAML file")
        
        return output_path


def main():
    """Main entry point for standalone execution."""
    print("Session YAML Generator")
    print("=" * 50)
    
    # Create generator with default paths
    generator = SessionYAMLGenerator()
    
    # Run the generation process
    output_path = generator.run()
    
    if output_path:
        print(f"\nTo use the generated configuration:")
        print(f"1. Review the generated file: {output_path}")
        print(f"2. Copy relevant sections to your main config.yaml")
        print(f"3. Add additional subject metadata as needed")
    
    return output_path


if __name__ == "__main__":
    main()
