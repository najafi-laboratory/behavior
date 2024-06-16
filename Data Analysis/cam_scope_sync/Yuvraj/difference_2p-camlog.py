import pandas as pd
import plotly.express as px

df = pd.read_excel("difference_camlog_2p.xlsx")

fig = px.line(df, y='difference_camlog-2p', title='VG01 20240517 Camlog-2P per Frame')
fig.update_xaxes(title_text='Frames')
fig.update_yaxes(title_text='Time (Milliseconds)')

fig.show()
fig.write_html("difference_2p-camlog.html")