import os
from tqdm import tqdm, trange
# from phi.flow import *
from phi.torch.flow import *
import numpy as np
from utils.visualizations import create_plot
from utils.visualizations import curl
import hydra
from utils.generation import IForce
import copy


def step(velocity: StaggeredGrid, pressure: StaggeredGrid, force: IForce, args, dt: float = 1.0):
    velocity = advect.mac_cormack(velocity, velocity, dt, integrator=advect.rk4)
    if args.force.amplitude > 0:
        f = force() @ velocity
        velocity += f
    if args.alphav > 0:
        velocity += (- args.alphav) * velocity
    compressibility_solver = math.Solve('auto', args.incompressibility.precision, 0,
                                        max_iterations=args.incompressibility.max_iterations,
                                        gradient_solve=math.Solve('auto', args.incompressibility.precision, 0,
                                                                  max_iterations=args.incompressibility.max_iterations))
    velocity, pressure = fluid.make_incompressible(velocity, solve=compressibility_solver)
    velocity = diffuse.explicit(velocity, args.nu, dt)
    return velocity, pressure


@hydra.main(config_path='configs', config_name='navier2dgenerate')
def main(args):
    print(args)
    TORCH.set_default_device('GPU')
    TORCH.seed(args.seed)
    np.random.seed(args.seed)
    force = hydra.utils.instantiate(args.force)
    initial_v_xs = []
    initial_v_ys = []
    final_v_xs = []
    final_v_ys = []
    path = 'temp'
    if args.alphav > 0:
        path = f'{path}_av-{args.alphav}'
    os.makedirs(os.path.join('vorticity', path), exist_ok=True)
    os.makedirs(os.path.join('correlation', path), exist_ok=True)
    os.makedirs(os.path.join('temp', path), exist_ok=True)
    for i in trange(1100):
        velocity = StaggeredGrid(Noise(), extrapolation.PERIODIC, x=args.resolution, y=args.resolution,
                                 bounds=Box[0:1, 0:1])
        pressure = None
        for time_step in trange(args.time_steps):
            velocity, pressure = step(velocity, pressure, force, args, dt=args.dt)
            if time_step == 0:
                initial_v_xs.append(velocity.values.vector[0].numpy('y,x'))
                initial_v_ys.append(velocity.values.vector[1].numpy('y,x'))
            if i == 0:
                vorticity = curl(velocity.values.vector[0].numpy('y,x'), velocity.values.vector[1].numpy('y,x'),
                                 velocity.values.vector[0].numpy('y,x').shape[1])

                create_plot(f'vorticity/{path}/vorticity_{i:03d}_{time_step:05d}.png', vorticity,
                            bounds=0.1)

        final_v_xs.append(velocity.values.vector[0].numpy('y,x'))
        final_v_ys.append(velocity.values.vector[1].numpy('y,x'))

    save_path = hydra.utils.get_original_cwd()
    if args.force.amplitude > 0:
        save_path = os.path.join(save_path,
                                 f'data/navier2d{args.resolution}_{args.force.freq}_{args.force.amplitude}_{int(args.time_steps * args.dt):03d}.npz')
    else:
        save_path = os.path.join(save_path,
                                 f'data/navier2d{args.resolution}_{int(args.time_steps * args.dt):03d}.npz')
    np.savez(save_path,
             initial_v_x=np.stack(initial_v_xs, 0), initial_v_y=np.stack(initial_v_ys, 0),
             final_v_x=np.stack(final_v_xs, 0),
             final_v_y=np.stack(final_v_ys, 0), args=args)

    # Interpolation data
    samples = []
    for i in trange(50):
        cur_sample = []
        pressure = None
        velocity = StaggeredGrid(Noise(), extrapolation.PERIODIC, x=args.resolution, y=args.resolution,
                                 bounds=Box[0:1, 0:1])
        for time_step in trange(200):
            velocity, pressure = step(velocity, pressure, force, args, dt=args.interpolate_dt)
            cur_sample.append(
                np.stack([copy.deepcopy(velocity.values.vector[0].numpy('y,x')),
                          copy.deepcopy(velocity.values.vector[1].numpy('y,x'))], -1))

        samples.append(np.stack(cur_sample, 0))

    samples = np.stack(samples, 0)

    save_path = hydra.utils.get_original_cwd()
    if args.force.amplitude > 0:
        save_path = os.path.join(save_path,
                                 f'data/navier2dinterpolation{args.resolution}_{args.force.freq}_{args.force.amplitude}_{int(args.time_steps * args.dt):03d}.npz')
    else:
        save_path = os.path.join(save_path,
                                 f'data/navier2dinterpolation{args.resolution}_{int(args.time_steps * args.dt):03d}.npz')
    np.savez(save_path, samples=samples)


if __name__ == '__main__':
    main()
