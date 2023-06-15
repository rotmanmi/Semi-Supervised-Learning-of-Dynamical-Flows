from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field, InitVar
import numpy as np
from phi.torch.flow import *


def grid_sample(resolution: Tuple[int], scale: float, k_min: int, k_max: int, smoothness: float):
    rndj = (np.random.randn(*resolution) + + 1j * np.random.randn(*resolution))

    kaxis = np.meshgrid(*[np.fft.fftfreq(r) for r in resolution])
    k = np.sum((np.stack(kaxis, -1) * scale * resolution[0]) ** 2, -1)
    lowest_frequency = 0.1
    weight_mask = (k > lowest_frequency) & (k > k_min) & (k < k_max)
    k[(0,) * len(resolution)] = np.inf
    inv_k = 1 / k
    inv_k[(0,) * len(k.shape)] = 0

    fft = rndj * inv_k ** smoothness * weight_mask
    array = np.real(np.fft.ifftn(fft, axes=[i for i in range(len(resolution))]))
    array /= np.std(array)
    array -= np.mean(array)
    return array


@dataclass
class IForce(ABC):
    name: str

    @abstractmethod
    def __call__(self):
        ...


@dataclass
class CosForce(IForce):
    freq: int
    amplitude: float
    resolution: int

    def __call__(self) -> Grid:
        freq = 2 * np.pi * self.freq
        x = np.linspace(0, 1 - 1.0 / self.resolution, self.resolution)
        y = np.linspace(0, 1 - 1.0 / self.resolution, self.resolution)
        xv, yv = np.meshgrid(x, y)
        cosine_force = np.cos(freq * yv)
        force = self.amplitude * (StaggeredGrid(
            math.tensor(cosine_force, math.Shape(sizes=[self.resolution, self.resolution], names=['x', 'y'],
                                                 types=['spatial', 'spatial'])),
            extrapolation.PERIODIC,
            x=self.resolution, y=self.resolution) * (1, 0))
        return force


@dataclass
class RandomForce(IForce):
    k_min: int
    k_max: int
    amplitude: float
    resolution: int
    smoothness: float

    def __call__(self) -> Grid:
        random_noise = grid_sample([self.resolution, self.resolution],
                                   self.amplitude,
                                   self.k_min,
                                   self.k_max,
                                   self.smoothness)
        force_tensor = math.tensor(random_noise, math.Shape(sizes=[self.resolution, self.resolution], names=['x', 'y'],
                                                            types=['spatial', 'spatial']))
        force = StaggeredGrid(force_tensor, extrapolation.PERIODIC, x=self.resolution, y=self.resolution,
                              bounds=Box[0:1, 0:1])
        return force


@dataclass
class NoForce(IForce):
    amplitude: float
    resolution: int

    def __call__(self) -> int:
        return 0
