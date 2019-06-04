import numpy as np
from scipy import integrate

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import sys

from neuron_movie_lib import (
    load_neuron,
    set_neuron_axes,
)

N_trajectories = int(sys.argv[1])

plt.rcParams["font.family"] = "arial"
fontsize = 14

if (N_trajectories == 1):
    N_str = 'one'
elif (N_trajectories == 2):
    N_str = 'two'
elif (N_trajectories == 3):
    N_str = 'three'

T = 100
stop_T = 50
t_end = 0.8
dt = t_end / T
t = np.linspace(0, t_end, T)

# traj 1
x1_x, times = load_neuron(1, t)
# traj 2
x2_x, _ = load_neuron(2, t)

Writer = animation.writers['ffmpeg']
print(Writer)
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
print(writer)

# Solve for the trajectories
x_1 = np.array([t, 
                x1_x,
                np.cos(2*np.pi*t),
                ])
x_2 = np.array([t,
                x2_x,
                np.sin(4*np.pi*t)])
x_t = np.stack((x_1, x_2), 0)

x_t = x_t[:N_trajectories]
# add extra points of chilling
end_point = np.expand_dims(x_t[:,:,-1], 2)
x_t = np.concatenate((x_t, np.tile(end_point, [1, 1, stop_T])), 2)
total_T = T + stop_T

# Set up figure & 3D axis for animation
#ax = fig.add_axes([0, 0, 1, 1], label='label?')
if (N_trajectories == 1):
    fig, axs = plt.subplots(2, 1, figsize = (8, 4), gridspec_kw={'height_ratios': [1, 3]})
    ax = axs[0]
    ax.eventplot(times, colors=['k'], orientation='horizontal', linewidths=2);
    ax.axis('off');
    ax = axs[1]
    set_neuron_axes(ax, t_end, 1, fontsize=fontsize)
    plt.tight_layout()
elif (N_trajectories == 2):
    fig, axs = plt.subplots(2, 1, figsize = (8, 4))
    set_neuron_axes(axs[0], t_end, 1, fontsize=fontsize)
    set_neuron_axes(axs[1], t_end, 2, fontsize=fontsize)
    plt.tight_layout()
else:
    fig = plt.figure(figsize = (8, 4))
    ax = plt.gca()


# choose a different color for each trajectory
colors = [[0.0, 0.0, 0.8],
          [0.0, 0.0, 0.8]]

# set up lines and points
for i in range(N_trajectories):
    if (N_trajectories==1):
        ind = 1
    else:
        ind = i
    if i == 0:
        lines = axs[ind].plot([], [], '-', c=colors[i])
        pts = axs[ind].plot([], [], 'o', c=colors[i])
    else:
        lines += axs[ind].plot([], [], '-', c=colors[i])
        pts += axs[ind].plot([], [], 'o', c=colors[i])

# initialization function: plot the background of each frame
def init():
    for line, pt in zip(lines, pts):
        line.set_data([], [])

        pt.set_data([], [])
    return lines + pts


step = 2
# animation function.  This will be called sequentially with the frame number
def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (step * i) % x_t.shape[2]

    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z = xi
        line.set_data(x, y)

        pt.set_data(x[i], y[i])

    fig.canvas.draw()
    return lines + pts

# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=total_T//step, interval=30, blit=True)

# Save as mp4. This requires mplayer or ffmpeg to be installed
print('Making video.')
anim.save('%s_neurons.mp4' % N_str, writer=writer)
print('Video complete.')
#plt.show()
