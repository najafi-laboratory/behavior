import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parameters import FILE_PATH, MASK_VALUE, FRAME_RATE


def calculate_mean(dataset):
    center_pupil_x = dataset['center_pupil_x'].mean()
    center_pupil_y = dataset['center_pupil_y'].mean()
    return center_pupil_x, center_pupil_y


def form_dataset():
    data = pd.read_csv(FILE_PATH, header=[1, 2])
    data.columns = [f'{i}_{j}' if isinstance(j, str) else f'{i}' for i, j in data.columns]
    likelihood_col = 'center_pupil_likelihood'
    mask = (data[likelihood_col] >= MASK_VALUE)
    dataset = data[mask].copy()
    mean_x, mean_y = calculate_mean(dataset)
    return mean_x, mean_y, dataset


def pupil_tracking_over_time():
    mean_x, mean_y, dataset = form_dataset()
    dataset['distance'] = np.sqrt((dataset['center_pupil_x'] - mean_x)**2 + (dataset['center_pupil_y'] - mean_y)**2)
    time_k = np.arange(len(dataset)) / FRAME_RATE
    return time_k, dataset


time, filtered_data = pupil_tracking_over_time()
plt.figure(figsize=(10, 6))
plt.plot(time, filtered_data['distance'], linestyle='-', color='purple')
plt.title('Deviation of Center Pupil Position From Mean Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Distance from Mean Position (pixels)')
plt.show()
