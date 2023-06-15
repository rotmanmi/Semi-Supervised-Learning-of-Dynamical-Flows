import scipy.signal as signal
import math
import torch
import numpy as np
import matplotlib.pyplot as plt


def curl(u, v, length):
    # length is the box length, 128 in your case
    if isinstance(u, torch.Tensor):
        u = u.cpu().detach().numpy()
    if isinstance(v, torch.Tensor):
        v = v.cpu().detach().numpy()
    res = u.shape[0]
    xaxis = np.linspace(0, length, res)
    yaxis = np.linspace(0, length, res)

    kerndx = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
    kerndy = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])

    dx = (xaxis[2] - xaxis[0])
    dy = (yaxis[2] - yaxis[0])

    u = np.concatenate((u[-1:, :], u, u[0:1, :]), axis=0)
    u = np.concatenate((u[:, -1:], u, u[:, 0:1]), axis=1)
    v = np.concatenate((v[-1:, :], v, v[0:1, :]), axis=0)
    v = np.concatenate((v[:, -1:], v, v[:, 0:1]), axis=1)

    dudy = signal.convolve(u, kerndy, mode='valid') / dy
    dvdx = signal.convolve(v, kerndx, mode='valid') / dx
    curlz = dvdx - dudy

    return curlz


def create_plot(path: str, field_x: np.ndarray, bounds: float = 1.5):
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=-bounds, vmax=bounds)

    plt.imsave(path, cmap(norm(np.tile(field_x, (2, 2)))))
