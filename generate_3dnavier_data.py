import os
from tqdm import tqdm, trange
from phi.flow import *
# from phi.torch.flow import *
import pylab
import matplotlib.pyplot as plt
import numpy as np

DT = 0.1
NU = 0.001
np.random.seed(1234)


def step(velocity, pressure, dt=1.0, buoyancy_factor=0.1):
    velocity = advect.semi_lagrangian(velocity, velocity, dt)

    velocity = diffuse.explicit(velocity, NU, dt)
    velocity, pressure = fluid.make_incompressible(velocity)
    return velocity, pressure


initial_v_xs = []
initial_v_ys = []
initial_v_zs = []
final_v_xs = []
final_v_ys = []
final_v_zs = []
os.makedirs('temp', exist_ok=True)
for i in trange(1100):
    velocity = StaggeredGrid(Noise(), extrapolation.PERIODIC, x=64, y=64, z=64, bounds=Box[0:128, 0:128, 0:128])
    pressure = None
    for time_step in trange(10):
        velocity, pressure = step(velocity, pressure, dt=DT)

        if time_step == 0:
            initial_v_xs.append(velocity.values.vector[0].numpy('z,y,x'))
            initial_v_ys.append(velocity.values.vector[1].numpy('z,y,x'))
            initial_v_zs.append(velocity.values.vector[2].numpy('z,y,x'))

    final_v_xs.append(velocity.values.vector[0].numpy('z,y,x'))
    final_v_ys.append(velocity.values.vector[1].numpy('z,y,x'))
    final_v_zs.append(velocity.values.vector[2].numpy('z,y,x'))

np.savez('data/navier3d.npz',
         initial_v_x=np.stack(initial_v_xs, 0),
         initial_v_y=np.stack(initial_v_ys, 0),
         initial_v_z=np.stack(initial_v_zs, 0),
         final_v_x=np.stack(final_v_xs, 0),
         final_v_y=np.stack(final_v_ys, 0),
         final_v_z=np.stack(final_v_zs, 0))
