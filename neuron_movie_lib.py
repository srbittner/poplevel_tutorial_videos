import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

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
    if (n==1):
        np.random.seed(1)
        r = 10.0*(np.cos(4*np.pi*t) + 1.5)*dt
    elif (n==2):
        np.random.seed(0)
        r = 10.0*(np.sin(5*np.pi*t) + 1.5)*dt
    elif (n==3):
        r = 5.0*(np.sin(2*np.pi*t) + 1.5)*dt
    else:
        raise NotImplementedError()
    s = poisson_spikes(r)
    times = dt*np.where(s==1.0)[0]
    x = scipy.ndimage.gaussian_filter(s, 8.0)/dt
    return x, times

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
    ax.set_ylim((0, 45))
    return None

def set_ss_axes(ax, fontsize=12):
    ax.set_xlim((0, 45))
    ax.set_ylim((0, 45))
    ax.set_xlabel('neuron 1 \n rate', rotation=0, fontsize=fontsize)
    ax.set_ylabel('neuron 2 \n rate', rotation=0, fontsize=fontsize)
    ax.yaxis.set_label_coords(-0.2,0.6)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return None
