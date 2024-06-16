# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 21:41:34 2024

@author: gtg424h
"""


def find_onset_frames(etl_list, threshold=1):
    onset_indices = []
    recording = False
    
    for i, voltage in enumerate(etl_list):
        if voltage >= threshold and not recording:
            onset_indices.append(i)
            recording = True
            
        elif voltage < threshold and recording:
            recording = False
            
    return onset_indices

onset_frames = find_onset_frames(etl_list)
print(onset_frames)





# Create a boolean array where True indicates the voltage is above the threshold
above_threshold = np.array(etl_list) > 1

# Find the difference between consecutive elements in the boolean array
# The onset corresponds to a change from False to True
onset_indices = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1

len(onset_indices)
print(onset_indices)



from scipy.interpolate import interp1d

def upsample_array(array, factor=2000):
    original_length = len(array)
    new_length = original_length * factor

    # Original indices
    original_indices = np.arange(original_length)

    # New indices
    new_indices = np.linspace(0, original_length - 1, new_length)

    # Interpolation function
    interp_func = interp1d(original_indices, array, kind='linear')

    # Upsampled array
    upsampled_array = interp_func(new_indices)

    return upsampled_array


cu = upsample_array(c)





import numpy as np

def find_onsets_offsets(etl_list, threshold=1):
    above_threshold = etl_list > threshold
    onset_indices = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1
    offset_indices = np.where(np.diff(above_threshold.astype(int)) == -1)[0] + 1
    return onset_indices, offset_indices

def downsample_with_events(etl_list, original_rate=2000, target_rate=1000, threshold=1):
    # Find onsets and offsets
    onset_indices, offset_indices = find_onsets_offsets(etl_list, threshold)

    # Calculate downsample factor
    factor = original_rate // target_rate
    
    # Generate regular downsampled indices
    downsampled_indices = np.arange(0, len(etl_list), factor)

    # Combine downsampled indices with onset and offset indices
    key_indices = np.unique(np.concatenate((downsampled_indices, onset_indices, offset_indices)))
    
    # Sort the indices to maintain order
    key_indices.sort()

    # Downsample the array using the key indices
    downsampled_array = etl_list[key_indices]
    
    return downsampled_array, key_indices

# Example usage
etl_list = np.array(etl_list)
downsampled_array, key_indices = downsample_with_events(etl_list)

print("Original array length:", len(etl_list))
print("Downsampled array length:", len(downsampled_array))
print("Downsampled array:", downsampled_array)
print("Key indices:", key_indices)



onset_indicesd, offset_indicesd = find_onsets_offsets(downsampled_array, threshold)


# Create a boolean array where True indicates the voltage is above the threshold
above_threshold = np.array(downsampled_array) > 1

# Find the difference between consecutive elements in the boolean array
# The onset corresponds to a change from False to True
onset_indices1 = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1

print(onset_indices1)






