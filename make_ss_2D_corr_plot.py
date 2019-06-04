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

N_trajectories = 2

plt.rcParams["font.family"] = "arial"
fontsize = 10

N_str = 'two'

T = 100
t_end = 0.8
dt = t_end / T
t = np.linspace(0, t_end, T)
# traj 1
x1, _ = load_neuron(1, t)

# traj 2
x2, _ = load_neuron(2, t)

inds = range(0, T, skip)
x1 = x1[inds]
x2 = x2[inds]

fig = plt.figure(figsize=(4, 4))
ax = plt.gca()
set_ss_axes(ax, fontsize)
plt.tight_layout()

# choose a different color for each trajectory
color = [0.0, 0.0, 0.8]

ax.plot(x1, x2, 'o', c=color)

for nstd in [1, 2]:
    cov = np.cov(x1, x2) 
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    theta = np.degrees(np.arctan2(*v[:,0][::-1]))
    #theta = np.rad2deg(np.arccos(v[0, 0]))
    ell = Ellipse(xy=(np.mean(x1), np.mean(x2)),
                      width=2*nstd*lambda_[0], height=2*nstd*lambda_[1],
                      angle=theta, color='black')
    ell.set_facecolor('none')
    ax.add_artist(ell)

plt.savefig('2D_corr.png')
plt.show()
