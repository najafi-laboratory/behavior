import xml.etree.ElementTree as ET
import os
import numpy as np


def explore_xml_structure(xml_file_path, max_elements_to_show=5):
    """
    Explore the XML structure to understand the format.
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        print(f"Root element: {root.tag}")
        print(f"Root attributes: {root.attrib}")
        
        def print_element_structure(element, level=0, max_level=3):
            indent = "  " * level
            if level <= max_level:
                print(f"{indent}{element.tag}: {element.attrib}")
                if element.text and element.text.strip():
                    print(f"{indent}  Text: {element.text.strip()}")
                
                # Show first few children
                children = list(element)
                if children:
                    shown = 0
                    for child in children:
                        if shown < max_elements_to_show:
                            print_element_structure(child, level + 1, max_level)
                            shown += 1
                        else:
                            print(f"{indent}  ... and {len(children) - shown} more {child.tag} elements")
                            break
        
        print("\n=== XML STRUCTURE ===")
        print_element_structure(root)
        
        # Look specifically for Frame elements and their structure
        frames = root.findall('.//Frame')
        print(f"\n=== FRAME ANALYSIS ===")
        print(f"Found {len(frames)} Frame elements")
        
        if frames:
            print(f"\nFirst frame structure:")
            first_frame = frames[0]
            print(f"Frame tag: {first_frame.tag}")
            print(f"Frame attributes: {first_frame.attrib}")
            
            # Show all children of first frame
            for child in first_frame:
                print(f"  Child: {child.tag} = {child.text} (attrib: {child.attrib})")
        
        return True
        
    except Exception as e:
        print(f"Error exploring XML: {e}")
        return False


def extract_frame_timing_vector(xml_file_path):
    """
    Extract vector of relative frame times from PrairieView XML file.
    
    Args:
        xml_file_path: Path to the XML file
        
    Returns:
        dict: Contains frame timing vector and metadata
    """
    try:
        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        print(f"Loading XML file: {xml_file_path}")
        print(f"Root element: {root.tag}")
        
        # Find all Frame elements
        frames = root.findall('.//Frame')
        print(f"Found {len(frames)} frames")
        
        if not frames:
            return {"error": "No Frame elements found in XML"}
        
        # Extract timing information from all frames
        frame_data = []
        
        for frame in frames:
            # Try different possible timing element names
            relative_time = None
            frame_index = None
            
            # Get timing information
            relative_time_elem = frame.find('relativeTime')
            if relative_time_elem is not None:
                relative_time = float(relative_time_elem.text)
            else:
                # Try 'absoluteTime'
                absolute_time_elem = frame.find('absoluteTime')
                if absolute_time_elem is not None:
                    relative_time = float(absolute_time_elem.text)
                else:
                    # Try as attribute
                    if 'relativeTime' in frame.attrib:
                        relative_time = float(frame.attrib['relativeTime'])
                    elif 'absoluteTime' in frame.attrib:
                        relative_time = float(frame.attrib['absoluteTime'])
                    elif 'time' in frame.attrib:
                        relative_time = float(frame.attrib['time'])
                    else:
                        # Try other common time element names
                        time_elem = frame.find('time')
                        if time_elem is not None:
                            relative_time = float(time_elem.text)
            
            # Get frame index
            index_elem = frame.find('index')
            if index_elem is not None:
                frame_index = int(index_elem.text)
            elif 'index' in frame.attrib:
                frame_index = int(frame.attrib['index'])
            
            # Store frame data if we have timing info
            if relative_time is not None:
                frame_data.append({
                    'index': frame_index,
                    'relative_time': relative_time
                })
        
        if not frame_data:
            return {"error": "No timing information found in frames"}
        
        # Sort by frame index if available, otherwise by relative time
        if all(frame['index'] is not None for frame in frame_data):
            frame_data.sort(key=lambda x: x['index'])
            print(f"Sorted frames by index")
        else:
            frame_data.sort(key=lambda x: x['relative_time'])
            print(f"Sorted frames by relative time")
        
        # Extract timing vector
        relative_times = np.array([frame['relative_time'] for frame in frame_data])
        frame_indices = np.array([frame['index'] for frame in frame_data if frame['index'] is not None])
        
        # Calculate frame intervals and statistics
        if len(relative_times) > 1:
            intervals = np.diff(relative_times)
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            estimated_frame_rate = 1.0 / mean_interval if mean_interval > 0 else 0
        else:
            intervals = np.array([])
            mean_interval = 0
            std_interval = 0
            estimated_frame_rate = 0
        
        # Get additional metadata
        metadata = {}
        
        # Try to find version info
        version_elem = root.find('.//version')
        if version_elem is not None:
            metadata['prairie_version'] = version_elem.text
        
        # Try to find date
        date_elem = root.find('.//date')
        if date_elem is not None:
            metadata['acquisition_date'] = date_elem.text
        
        # Try to find TSeries info
        tseries_elem = root.find('.//TSeries')
        if tseries_elem is not None:
            # Look for timing information in TSeries
            tseries_time = tseries_elem.find('.//time')
            if tseries_time is not None:
                metadata['tseries_time'] = tseries_time.text
        
        results = {
            "success": True,
            "xml_file": xml_file_path,
            "relative_times": relative_times,  # The timing vector you requested
            "frame_indices": frame_indices if len(frame_indices) > 0 else None,
            "intervals": intervals,
            "total_frames": len(relative_times),
            "first_frame_time": relative_times[0],
            "last_frame_time": relative_times[-1],
            "total_duration_seconds": relative_times[-1] - relative_times[0],
            "mean_frame_interval": mean_interval,
            "std_frame_interval": std_interval,
            "estimated_frame_rate_hz": estimated_frame_rate,
            "metadata": metadata
        }
        
        # Print summary
        print("\n=== PRAIRIE XML TIMING VECTOR RESULTS ===")
        print(f"Total frames: {results['total_frames']}")
        print(f"First frame time: {results['first_frame_time']:.6f} seconds")
        print(f"Last frame time: {results['last_frame_time']:.6f} seconds")
        print(f"Total duration: {results['total_duration_seconds']:.6f} seconds")
        print(f"Mean frame interval: {results['mean_frame_interval']:.6f} seconds")
        print(f"Frame rate: {results['estimated_frame_rate_hz']:.2f} Hz")
        print(f"Frame interval std: {results['std_frame_interval']:.6f} seconds")
        
        if frame_indices is not None and len(frame_indices) > 0:
            print(f"Frame indices range: {frame_indices[0]} to {frame_indices[-1]}")
        
        if metadata:
            print("\nMetadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        
        # Show first few and last few timing values
        print(f"\nFirst 5 frame times: {relative_times[:5]}")
        print(f"Last 5 frame times: {relative_times[-5:]}")
        
        return results
        
    except ET.ParseError as e:
        return {"error": f"XML parsing error: {e}"}
    except FileNotFoundError:
        return {"error": f"File not found: {xml_file_path}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


def save_timing_vector(results, output_file=None):
    """
    Save the timing vector to a file.
    
    Args:
        results: Results dictionary from extract_frame_timing_vector
        output_file: Optional output file path (default: same directory as XML)
    """
    if not results.get("success"):
        print(f"Cannot save: {results.get('error', 'No results')}")
        return False
    
    try:
        if output_file is None:
            # Create output file name based on XML file
            xml_path = results['xml_file']
            base_name = os.path.splitext(xml_path)[0]
            output_file = f"{base_name}_frame_times.npy"
        
        # Save the timing vector
        np.save(output_file, results['relative_times'])
        print(f"Saved timing vector to: {output_file}")
        
        # Also save metadata as text
        metadata_file = output_file.replace('.npy', '_metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write(f"Frame Timing Analysis Results\n")
            f.write(f"XML File: {results['xml_file']}\n")
            f.write(f"Total Frames: {results['total_frames']}\n")
            f.write(f"Duration: {results['total_duration_seconds']:.6f} seconds\n")
            f.write(f"Frame Rate: {results['estimated_frame_rate_hz']:.2f} Hz\n")
            f.write(f"Mean Interval: {results['mean_frame_interval']:.6f} seconds\n")
            f.write(f"Std Interval: {results['std_frame_interval']:.6f} seconds\n")
            
            if results['metadata']:
                f.write(f"\nMetadata:\n")
                for key, value in results['metadata'].items():
                    f.write(f"  {key}: {value}\n")
        
        print(f"Saved metadata to: {metadata_file}")
        return True
        
    except Exception as e:
        print(f"Error saving timing vector: {e}")
        return False


# Keep the old function for backward compatibility
load_prairie_xml_and_get_last_frame_time = extract_frame_timing_vector


if __name__ == "__main__":
    # Path to the XML file
    xml_file_path = r"D:\behavior\2p_imaging\processed\2afc\YH24LG\YH24LG_CRBL_lobulev_20250620_2afc-494\YH24LG_CRBL_lobulev_20250620_2afc-494.xml"
    
    # Check if file exists
    if not os.path.exists(xml_file_path):
        print(f"ERROR: XML file not found at {xml_file_path}")
    else:
        # First explore the structure
        print("=== EXPLORING XML STRUCTURE ===")
        explore_xml_structure(xml_file_path)
        
        print("\n" + "="*50)
        
        # Extract the timing vector
        results = extract_frame_timing_vector(xml_file_path)
        
        if results.get("success"):
            print(f"\n*** TIMING VECTOR EXTRACTED ***")
            print(f"Vector shape: {results['relative_times'].shape}")
            print(f"Last frame time: {results['last_frame_time']:.6f} seconds")
            
            # Save the timing vector
            save_timing_vector(results)
            
        else:
            print(f"ERROR: {results.get('error', 'Unknown error')}")