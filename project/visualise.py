import plotly.graph_objs as go
import numpy as np

# Generate random 3D points

# xs = np.random.randn(100)
# ys = np.random.randn(100)
# zs = np.random.randn(100)

cs,xs,ys,zs = np.load("point_cloud.npy")
np.random.seed(0)
n_points = 100

# Create a trace
trace = go.Scatter3d(
    x=xs,
    y=ys,
    z=zs,
    mode='markers',
    marker=dict(
        size=2,    
        color = zs,# set color to an array/list of desired values
        opacity=0.4
    )
)

# Create layout
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='X Axis',visible=False),
        yaxis=dict(title='Y Axis',visible=False),
        zaxis=dict(range = [0,2],title='Z Axis',visible=False),
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

# Create a figure
fig = go.Figure(data=[trace], layout=layout)
# fig.update_xaxes(visible=False)
# fig.update_yaxes(visible=False)

# Plot the figure
fig.show()
