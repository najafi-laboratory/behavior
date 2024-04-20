DLC Output Grapher

Last Updated: April 20th, 2024

Codes to create graphs on the .csv files created by the DLC output.

Prerequisites:
- Python 3.8 or higher
- Pandas library
- NumPy library
- Matplotlib library

Keynotes:
1) Make sure to adjust parameter tables to fit specific videos.
2) To run each file, go to that Python file, each creates a different graph.
3) Everything was made into functions to be modular; i.e. can move things around to create other graphs. Some graphs
dependencies on others, so import ALL the graphs to work.


TODO:
- fix units for pixels, appears to calculate subpixel size.


Guide to each graph/file:
1) Parameters: contains constants that need to be calibrated per graph.

2) Center Pupil over time: plots the deviation from the mean of the center pupil over time.

3) Center Pupil over timeXY: plots 2 line graphs representative of the x and y median from the center pupil over time.

4) Center Pupil likelihood: tracks the x-y coordinates of the central pupil throughout the graph. Color represents the
confidence of the model (does not use MASK)

5) Polygon Area Calculator: calculates and plots the area of a polygon formed by specified points (ventral, dorsal,
nasal, and temporal) over time.
