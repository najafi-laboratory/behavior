import numpy as np
import matplotlib.pyplot as plt
from parameters import FRAME_RATE
from pupilTrackingOverTime import form_dataset


def pupil_tracking_over_time():
    mean_x, mean_y, dataset = form_dataset()
    dataset['x_distance'] = dataset['center_pupil_x'] - mean_x
    dataset['y_distance'] = dataset['center_pupil_y'] - mean_y
    time_k = np.arange(len(dataset)) / FRAME_RATE
    return time_k, dataset


time, filtered_data = pupil_tracking_over_time()

plt.figure(figsize=(10, 6))
plt.plot(time, filtered_data['x_distance'], linestyle='-', color='blue')
plt.title('Deviation of Center Pupil Position (X) From Mean Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('X Distance from Mean Position (pixels)')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, filtered_data['y_distance'], linestyle='-', color='red')
plt.title('Deviation of Center Pupil Position (Y) From Mean Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Y Distance from Mean Position (pixels)')
plt.show()
