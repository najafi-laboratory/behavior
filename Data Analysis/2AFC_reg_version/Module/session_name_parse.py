import os
import re

def parse_behavior_file_path(path):
    """
    Parses a behavioral data file path to extract:
    - Subject name (e.g., LCHR_TS01, SCHR_TS02, TCHR_TS03, YH24LG)
    - Version info (e.g., V_1_10, V_1_11)
    - Session date (e.g., 20250417, 20250523)

    Args:
        path (str): Full path to the .mat file

    Returns:
        tuple: (subject, version, session_date)

    Raises:
        ValueError: If the filename doesn't match the expected pattern or path is invalid
    """
    if not isinstance(path, str) or not path:
        raise ValueError("Path must be a non-empty string")

    # Get filename without extension
    name_without_ext = os.path.splitext(os.path.basename(path))[0]
    
    # Regex pattern for subject (alphanumeric with optional underscore), version, and date
    pattern = r"(?P<subject>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*).*?(?P<version>V_\d+_\d+).*?(?P<date>\d{8})"
    
    try:
        match = re.search(pattern, name_without_ext)
        if match:
            return (
                match.group("subject"),
                match.group("version"),
                match.group("date")
            )
        raise ValueError(f"Filename '{name_without_ext}' does not match expected pattern")
    except re.error as e:
        raise ValueError(f"Regex error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error parsing filename: {str(e)}")