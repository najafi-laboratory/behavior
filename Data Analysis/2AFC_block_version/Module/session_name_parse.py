import os
import re

def parse_behavior_file_path(path):
    """
    Parses a behavioral data file path to extract:
    - Subject name
    - Version info
    - Session date

    Parameters:
    - path (str): Full path to the .mat file

    Returns:
    - subject (str): e.g., "YH24LG"
    - version (str): e.g., "V_1"
    - session_date (str): e.g., "20250628"
    """
    # Get just the filename
    filename = os.path.basename(path)

    # Remove the extension
    name_without_ext = os.path.splitext(filename)[0]

    # Use regex to extract parts
    match = re.search(r"(?P<subject>[A-Za-z0-9]+)_block.*?(?P<version>V_\d+).*?(?P<date>\d{8})", name_without_ext)
    if match:
        subject = match.group("subject")
        version = match.group("version")
        session_date = match.group("date")
        return subject, version, session_date
    else:
        raise ValueError("Filename format did not match expected pattern.")