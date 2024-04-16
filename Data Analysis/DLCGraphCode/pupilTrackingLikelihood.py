import pandas as pd
import matplotlib.pyplot as plt
from parameters import FILE_PATH, get_color


def pupil_tracking_by_likelihood():
    data_k = pd.read_csv(FILE_PATH, header=[1, 2])
    data.columns = [f'{i}_{j}' if isinstance(j, str) else f'{i}' for i, j in data.columns]
    likelihood = 'center_pupil_likelihood'
    colors_k = get_color(data[likelihood])
    return data_k, colors_k


data, colors = pupil_tracking_by_likelihood()
plt.figure(figsize=(10, 6))
plt.scatter(data['center_pupil_x'], data['center_pupil_y'], color=colors, alpha=0.5)
plt.title(f'Center Pupil Spatial Coordinates by Likelihood')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()
