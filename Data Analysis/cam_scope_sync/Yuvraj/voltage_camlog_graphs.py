import pandas as pd
import plotly.graph_objects as go

data = pd.read_excel("voltage_camlog_frames_aligned.xlsx")

fig = go.Figure()

fig.add_trace(go.Scatter(x=list(range(len(data))), y=data["voltage"], mode="lines", name="Voltage"))
fig.add_trace(go.Scatter(x=list(range(len(data))), y=data["camlog"], mode="lines", name="Camlog"))
fig.add_trace(go.Scatter(x=list(range(len(data))), y=data["cv2"], mode="lines", name="30 FPS"))

fig.update_layout(title="Voltage, Camlog, CV2 Time per Frames", xaxis_title="Frames", yaxis_title="Time (Milliseconds)")

#fig.show()
fig.write_html("voltage_camlog_milliseconds_per_frame.html")