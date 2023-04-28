
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

#Generate some sample data

theta = np.random.uniform(0, 2*np.pi, 1000)

r = np.random.normal(1, 0.1, 1000)

x = r*np.cos(theta)
y = r*np.sin(theta)
z = np.random.normal(0, 0.1, 1000)


# Create a 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5,
                color=theta,                # set color to an array/list of desired values
                colorscale='plasma',
                cmin = 0,
                cmax = 2,  # choose a colorscale
                opacity=0.8,
                colorbar = dict(title="Angle", titleside="right", tickmode='auto', nticks=5, len=0.5,)))])

# # Create a 3D scatter plot
# fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5,
#                 color=theta,                # set color to an array/list of desired values
#                 colorscale='plasma',   # choose a colorscale
#                 opacity=0.8,
#                 ))])

# colorbar = dict(title='Values', tickmode='auto', nticks=5, len=0.5,
#                 tickfont=dict(size=10), thickness=10,
#                 xpad=0, ypad=0, bgcolor='white',
#                 tickvals=np.linspace(0, 1, 5), ticktext=np.linspace(0, 1, 5))

#Update the layout to include the colorbar
# fig.update_layout(coloraxis=dict(colorscale = "Viridis",colorbar=colorbar), title=dict(text='My 3D Point Cloud', font=dict(size=24),
#                   x=0.5))

fig.update_coloraxes(cmin = 0, cmax = 1)

# Set the layout properties
#fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

# Save the plot as an interactive HTML file
pio.write_html(fig, 'plot.html')

#import matplotlib.pyplot as plt

# import mpld3

# # Create the figure and the 3D axes
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the data as a 3D scatter plot
# ax.scatter(x, y, z)

# # Set the axis labels and the title
# # ax.set_xlabel('X')
# # ax.set_ylabel('Y')
# # ax.set_zlabel('Z')
# ax.set_title('3D Scatter Plot')

# # Convert the plot to an HTML format using mpld3
# html = mpld3.fig_to_html(fig)

# # Save the HTML to a file
# with open('plot.html', 'w') as f:
#     f.write(html)