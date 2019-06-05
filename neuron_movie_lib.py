import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from scipy.linalg import qr

def poisson_spikes(r):
    """Simulate poisson series with rate r at each bin."""
    T = r.shape[0]
    s = np.zeros((T,))
    for i in range(T):
        s[i] = float(np.random.poisson(r[i]))
    return s


def load_neuron(n, t):
    """ I found some random draws of spikes that look good for tutorial. """
    dt = t[1] - t[0]
    t_end = t[-1]

    beta = 2.0
    x1 = np.cos(2*np.pi*t/t_end)
    x2 = np.sin(4*np.pi*t/t_end)
    x3 = beta*np.sin(2*np.pi*t/t_end)
    X = np.array([x1, x2, x3])

    np.random.seed(9)
    Q, _ = qr(np.random.randn(3,3))
    X_proj = np.dot(Q, X)
    
    x = X_proj[n-1]
   
    # get spike times
    np.random.seed(1)
    FR_fac = 5
    shift = np.min(x)
    rates = FR_fac*(x - shift)
    s = poisson_spikes(rates*dt)
    times = dt*np.where(s==1.0)[0]
    #x_rec = scipy.ndimage.gaussian_filter(s, 8.0)/dt
    return rates, times

def set_neuron_axes(ax, t_end, n, fontsize=12):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the axes labels
    ax.set_xlabel('time (s)', fontsize=fontsize)
    ax.set_ylabel('neuron %d \n rate' % n, rotation=0, fontsize=fontsize)
    ax.yaxis.set_label_coords(-0.2,0.6)

    # prepare the axes limits
    t_buf = 0.1
    ax.set_xlim((0, t_end+t_buf))
    ax.set_xticks([0.0, 0.8])
    ax.set_yticks([0, 10, 20])
    ax.set_ylim((0, 22))
    ax.tick_params(direction='out')
    return None

def set_ss_axes(ax, fontsize=12, dim=2):
    ax.set_xlim((0, 22))
    ax.set_ylim((0, 22))
    if (dim == 2):
        ax.set_xlabel('neuron 1 \n rate', rotation=0, fontsize=fontsize)
        ax.set_ylabel('neuron 2 \n rate', rotation=0, fontsize=fontsize)
        ax.yaxis.set_label_coords(-0.4,0.6)
        ax.set_xticks([0, 10, 20])
        ax.set_yticks([0, 10, 20])
    if (dim == 3):
        ax.set_xlabel('neuron 1 \n rate', rotation=0, fontsize=fontsize)
        ax.set_ylabel('neuron 2 \n rate', rotation=0, fontsize=fontsize)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel('neuron 3 \n rate', rotation=90, fontsize=fontsize)
        #ax.yaxis.set_label_coords(-0.2,0.6)
        ax.set_zlim((0, 15))
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return None
