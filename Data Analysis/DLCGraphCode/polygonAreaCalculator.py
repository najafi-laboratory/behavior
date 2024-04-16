import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from parameters import FRAME_RATE, FILE_PATH, MASK_VALUE


def polygon_area(x_coords, y_coords):
    x = np.array(x_coords)
    y = np.array(y_coords)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def area_of_eye():
    data = pd.read_csv(FILE_PATH, header=[1, 2])
    data.columns = ['_'.join(col).strip() for col in data.columns.values]

    points = ["ventral_pupil", "dorsal_pupil", "temporal_pupil", "nasal_pupil"]
    x_cols = [f"{point}_x" for point in points]
    y_cols = [f"{point}_y" for point in points]
    likelihood_cols = [f"{point}_likelihood" for point in points]

    mask = (data[likelihood_cols] > MASK_VALUE).all(axis=1)
    filtered_data = data.loc[mask]
    area = filtered_data.apply(lambda row: polygon_area([row[col] for col in x_cols],
                                                         [row[col] for col in y_cols]), axis=1)
    time = filtered_data.index / FRAME_RATE
    return area, time


areas, time = area_of_eye()
plt.figure(figsize=(10, 6))
plt.plot(time, areas.values, marker='o', linestyle='-')
plt.title('Average Area of Pupil Configuration Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Area (in pixels squared)')
plt.grid(True)
plt.show()


