import pandas as pd
import plotly.express as px

df = pd.read_excel("voltage_camlog_frames_aligned.xlsx")

newList = []

for index, row in df.iterrows():
    newList.append(row["camlog"] - row["voltage"])

df["difference_camlog-2p"] = newList

fig = px.line(df, y='difference_camlog-2p', title='VG01 20240517 Camlog-2P per Frame')
fig.update_xaxes(title_text='Frames')
fig.update_yaxes(title_text='Time (Milliseconds)')

fig.show()

df.to_excel("difference_camlog_2p.xlsx")