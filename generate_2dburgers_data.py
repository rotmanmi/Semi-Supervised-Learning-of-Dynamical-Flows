import os

import numpy as np
import pde
from pde import FieldCollection, PDEBase, UnitGrid
from tqdm import tqdm, trange
import pickle
import datetime
import pytz
import ray


def kernel(xx, yy, l=1, sigma=1):
    snx = np.sin(np.abs(xx.reshape(-1, 1) - xx.reshape(1, -1)) / 2) ** 2
    sny = np.sin(np.abs(yy.reshape(-1, 1) - yy.reshape(1, -1)) / 2) ** 2
    return sigma ** 2 * np.exp(-2 * snx / l ** 2) * np.exp(-2 * sny / l ** 2)


def generate_random_states_engine(xmin, xmax, grid_size, l=0.6, sigma=1):
    grid_points = np.linspace(xmin, xmax, grid_size).reshape(-1, 1)
    x = np.meshgrid(grid_points, grid_points)[0].flatten()
    y = np.meshgrid(grid_points, grid_points)[1].flatten()
    cov = kernel(x, y, l, sigma)
    L_periodic = np.linalg.cholesky(cov + 1E-8 * np.eye(grid_size ** 2))
    return L_periodic


def generate_random_states(seed, L_periodic, grid_size):
    np.random.seed(seed)

    seeds = np.random.randint(0, 10e6, size=2)

    np.random.seed(seeds[0])
    fx = (L_periodic @ np.random.randn(grid_size ** 2)).reshape([grid_size, grid_size])
    np.random.seed(seeds[1])
    fy = (L_periodic @ np.random.randn(grid_size ** 2)).reshape([grid_size, grid_size])

    return fx, fy


class BurgersPDE2D(PDEBase):

    def __init__(self, nu=1, bc='natural'):
        """ initialize the class with a diffusivity and boundary conditions
        for the actual field and its second derivative """
        self.nu = nu
        self.bc = bc

    def evolution_rate(self, state, t=0):
        u, v = state

        u_t = - u * u.gradient(bc=self.bc)[0] - v * u.gradient(bc=self.bc)[1] + self.nu * u.laplace(bc=self.bc)
        v_t = - v * v.gradient(bc=self.bc)[1] - u * v.gradient(bc=self.bc)[0] + self.nu * v.laplace(bc=self.bc)

        return pde.FieldCollection([u_t, v_t])


def save_dataset(dataset):
    now = datetime.datetime.now(tz=pytz.timezone('Israel'))
    now_str = f'{now.year}_{now.month}_{now.day}_{now.hour}:{now.minute}'
    with open(f"{dataset['dataset_name']}_{now_str}.pickle", 'wb') as fp:
        pickle.dump(dataset, fp)


def load_dataset(path):
    with open(path, 'rb') as json_file:
        dataset = pickle.load(json_file)

    return dataset


@ray.remote
def propagate_state(init_state, nu, time_steps, dt):
    eq = BurgersPDE2D(nu=nu, bc=['periodic', 'periodic'])
    for j in range(dataset['n_time_steps']):
        result = eq.solve(init_state, t_range=time_steps, dt=dt)
        init_state = result

    return result.data


if __name__ == '__main__':
    ray.init(num_cpus=12)
    dataset = {}

    dataset['dataset_size'] = 1100
    dataset['dataset_name'] = 'burgers_2d'
    dataset['grid_lim_left'] = 0
    dataset['grid_lim_right'] = 2 * np.pi
    dataset['grid_size'] = 2 ** 6
    dataset['nu'] = 0.001
    dataset['dt'] = 1e-5
    dataset['time_step'] = 1
    dataset['n_time_steps'] = 1

    grid = pde.CartesianGrid(
        [[dataset['grid_lim_left'], dataset['grid_lim_right']], [dataset['grid_lim_left'], dataset['grid_lim_right']]],
        [dataset['grid_size'], dataset['grid_size']], periodic=[True, True])
    bc = ['periodic', 'periodic']
    eq = BurgersPDE2D(nu=dataset['nu'], bc=bc)

    L_periodic = generate_random_states_engine(0, dataset['grid_lim_right'], dataset['grid_size'])

    X = []
    y = []
    for i in tqdm(range(dataset['dataset_size']), desc='generating dataset'):
        func_x, func_y = generate_random_states(i, L_periodic, dataset['grid_size'])
        func_x, func_y = func_x / 6, func_y / 6
        func = np.concatenate(
            [func_x.reshape(1, func_x.shape[0], func_x.shape[1]), func_y.reshape(1, func_y.shape[0], func_y.shape[1])],
            axis=0)
        init_state = pde.VectorField(grid, func)
        X.append(init_state.data)
        state = pde.VectorField(grid)
        y.append(propagate_state.remote(init_state, dataset['nu'], dataset['time_step'], dataset['dt']))

    y = ray.get(y)
    X = np.stack(X, 0)
    y = np.stack(y, 0)
    os.makedirs('data', exist_ok=True)
    np.save('data/burgers_2d_X', np.moveaxis(X, 1, 3))
    np.save('data/burgers_2d_y', np.moveaxis(y, 1, 3))
