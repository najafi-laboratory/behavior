import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from parameters import FRAME_RATE, FILE_PATH, MASK_VALUE, FILTER
from scipy import stats


def polygon_area(x_coords, y_coords):
    """Calculate the area of a polygon given x and y coordinates."""
    x = np.array(x_coords)
    y = np.array(y_coords)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def area_of_eye():
    """Compute the area of the eye based on provided CSV data filtering by likelihood and Z-score."""
    data = pd.read_csv(FILE_PATH, header=[1, 2])
    data.columns = ['_'.join(col).strip() for col in data.columns.values]

    points = ["ventral_pupil", "dorsal_pupil", "temporal_pupil", "nasal_pupil"]
    x_cols = [f"{point}_x" for point in points]
    y_cols = [f"{point}_y" for point in points]
    likelihood_cols = [f"{point}_likelihood" for point in points]

    mask = (data[likelihood_cols] > MASK_VALUE).all(axis=1)
    filtered_data = data.loc[mask]

    if FILTER:
        z_scores = np.abs(stats.zscore(filtered_data[x_cols + y_cols]))
        filtered_data = filtered_data[(z_scores < 3).all(axis=1)]

    area = filtered_data.apply(lambda row: polygon_area([row[col] for col in x_cols],
                                                        [row[col] for col in y_cols]), axis=1)
    time = np.arange(len(area)) / FRAME_RATE
    return area, time


areas, time = area_of_eye()
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)
line, = ax.plot(time, areas, linestyle='-', color='blue')
ax.set_title('Average Area of Pupil Configuration Over Time')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Area (in pixels squared)')
ax.grid(True)
plt.show()
