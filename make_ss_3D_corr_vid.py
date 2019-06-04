import numpy as np
from scipy import integrate

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
from matplotlib.patches import Ellipse
import sys

from neuron_movie_lib import (
    load_neuron, 
    set_neuron_axes, 
    set_ss_axes,
)

skip = int(sys.argv[1])

N_trajectories = 3

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

plt.rcParams["font.family"] = "arial"
fontsize = 10

N_str = 'two'

T = 100
stop_T = 50
total_T = T + stop_T
t_end = 0.8
dt = t_end / T
t = np.linspace(0, t_end, T)
# traj 1
x1, _ = load_neuron(1, t)

# traj 2
x2, _ = load_neuron(2, t)

# traj 3
x3, _ = load_neuron(3, t)

inds = range(0, T, skip)
x1 = x1[inds]
x2 = x2[inds]
x3 = x3[inds]

fig = plt.figure(figsize=(4, 4))
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
set_ss_axes(ax, fontsize, dim=3)

# choose a different color for each trajectory
color = [0.0, 0.0, 0.8]
def init():
    lines = []
    pts = ax.plot(x1, x2, x3, 'o', c=color)
    return lines + pts

lines = []
pts = ax.plot(x1, x2, x3, 'o', c=color)

step = 2
def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (step * i) % total_T

    ax.view_init(12, 180 + 1.0 * i)
    fig.canvas.draw()
    return lines + pts

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=30, blit=True)

print('Making video.')
anim.save('3D_corr.mp4', writer=writer)
print('Video complete.')
#plt.show()
