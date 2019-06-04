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
    set_ss_axes,
)

N_trajectories = int(sys.argv[1])

plt.rcParams["font.family"] = "arial"
fontsize = 10

if (N_trajectories == 2):
    N_str = 'two'
elif (N_trajectories == 3):
    N_str = 'three'
else:
    raise NotImplementedError()

T = 100
stop_T = 50
t_end = 0.8
dt = t_end / T
t = np.linspace(0, t_end, T)
# traj 1
x1_x, _ = load_neuron(1, t)

# traj 2
x2_x, _ = load_neuron(2, t)

# traj 3
x3_x, _ = load_neuron(3, t)

Writer = animation.writers['ffmpeg']
print(Writer)
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
print(writer)

filler = np.zeros((T,))

# Solve for the trajectories
x_1 = np.array([t, 
                x1_x,
                filler])

x_2 = np.array([t,
                x2_x,
                filler])

x_3 = np.array([t,
                x3_x,
                filler])

x_ss = np.array([x1_x,
                 x2_x,
                 x3_x])

if (N_trajectories == 2):
    x_t = np.stack((x_1, x_2, x_ss), 0)
elif (N_trajectories == 3):
    x_t = np.stack((x_1, x_2, x_3, x_ss), 0)

x_t = x_t[:N_trajectories+1]

# add extra points of chilling
end_point = np.expand_dims(x_t[:,:,-1], 2)
x_t = np.concatenate((x_t, np.tile(end_point, [1, 1, stop_T])), 2)
total_T = T + stop_T

# Set up figure & 3D axis for animation
#ax = fig.add_axes([0, 0, 1, 1], label='label?')
if (N_trajectories == 2):
    fig = plt.figure(figsize = (8, 4))
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,3)
    ax3 = plt.subplot(1,2,2)
    set_neuron_axes(ax1, t_end, 1, fontsize=fontsize)
    set_neuron_axes(ax2, t_end, 2, fontsize=fontsize)
    set_ss_axes(ax3, fontsize=fontsize)
    axs = [ax1, ax2, ax3]
elif (N_trajectories == 3):
    fig = plt.figure(figsize = (8, 4))
    ax1 = plt.subplot(3,2,1)
    ax2 = plt.subplot(3,2,3)
    ax3 = plt.subplot(3,2,5)
    ax4 = plt.subplot(1,2,2)
    set_neuron_axes(ax1, t_end, 1, fontsize=fontsize)
    set_neuron_axes(ax2, t_end, 2, fontsize=fontsize)
    set_neuron_axes(ax3, t_end, 3, fontsize=fontsize)
    set_ss_axes(ax4, fontsize=fontsize)
    axs = [ax1, ax2, ax3, ax4]
else:
    raise NotImplementedError()


# choose a different color for each trajectory
colors = [[0.0, 0.0, 0.8],
          [0.0, 0.0, 0.8],
          [0.0, 0.0, 0.8],
          [0.0, 0.0, 0.8]]
#colors = plt.cm.jet(np.linspace(0, 1, 4))

# set up lines and points
for i in range(N_trajectories+1):
    if i == 0:
        lines = axs[i].plot([], [], '-', c=colors[i])
        pts = axs[i].plot([], [], 'o', c=colors[i])
    else:
        lines += axs[i].plot([], [], '-', c=colors[i])
        pts += axs[i].plot([], [], 'o', c=colors[i])

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
anim.save('%s_neuron_ss.mp4' % N_str, writer=writer)
print('Video complete.')
#plt.show()
