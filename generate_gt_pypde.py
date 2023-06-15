import torch
from utilities3 import MatReader
import numpy as np
import matplotlib.pyplot as plt
import pde
from pde import CartesianGrid, MemoryStorage, PDEBase, ScalarField, plot_kymograph, ScalarField
from tqdm import tqdm
import argparse
import os


# PDE classes
class BurgersPDE(PDEBase):

    def __init__(self, nu=1, bc='natural'):
        """ initialize the class with a diffusivity and boundary conditions
        for the actual field and its second derivative """
        self.nu = nu
        self.bc = bc

    def evolution_rate(self, state, t=0):
        """ numpy implementation of the evolution equation """
        return - state * state.gradient(bc=self.bc)[0] + self.nu * state.laplace(bc=self.bc)


class GeneralizedBurgers2PDE(PDEBase):

    def __init__(self, nu=1, bc='natural'):
        """ initialize the class with a diffusivity and boundary conditions
        for the actual field and its second derivative """
        self.nu = nu
        self.bc = bc

    def evolution_rate(self, state, t=0):
        """ numpy implementation of the evolution equation """
        return - state * state * state.gradient(bc=self.bc)[0] + self.nu * state.laplace(bc=self.bc)


class GeneralizedBurgers3PDE(PDEBase):

    def __init__(self, nu=1, bc='natural'):
        """ initialize the class with a diffusivity and boundary conditions
        for the actual field and its second derivative """
        self.nu = nu
        self.bc = bc

    def evolution_rate(self, state, t=0):
        """ numpy implementation of the evolution equation """
        return - state * state * state * state.gradient(bc=self.bc)[0] + self.nu * state.laplace(bc=self.bc)


class GeneralizedBurgers4PDE(PDEBase):

    def __init__(self, nu=1, bc='natural'):
        """ initialize the class with a diffusivity and boundary conditions
        for the actual field and its second derivative """
        self.nu = nu
        self.bc = bc

    def evolution_rate(self, state, t=0):
        """ numpy implementation of the evolution equation """
        return - state * state * state * state * state.gradient(bc=self.bc)[0] + self.nu * state.laplace(bc=self.bc)


class GeneralizedBurgers5PDE(PDEBase):

    def __init__(self, nu=1, bc='natural'):
        """ initialize the class with a diffusivity and boundary conditions
        for the actual field and its second derivative """
        self.nu = nu
        self.bc = bc

    def evolution_rate(self, state, t=0):
        """ numpy implementation of the evolution equation """
        return - state * state * state * state * state * state.gradient(bc=self.bc)[0] + self.nu * state.laplace(
            bc=self.bc)


class GeneralizedBurgers6PDE(PDEBase):

    def __init__(self, nu=1, bc='natural'):
        """ initialize the class with a diffusivity and boundary conditions
        for the actual field and its second derivative """
        self.nu = nu
        self.bc = bc

    def evolution_rate(self, state, t=0):
        """ numpy implementation of the evolution equation """
        return - state * state * state * state * state * state * state.gradient(bc=self.bc)[
            0] + self.nu * state.laplace(bc=self.bc)


class GeneralizedBurgers7PDE(PDEBase):

    def __init__(self, nu=1, bc='natural'):
        """ initialize the class with a diffusivity and boundary conditions
        for the actual field and its second derivative """
        self.nu = nu
        self.bc = bc

    def evolution_rate(self, state, t=0):
        """ numpy implementation of the evolution equation """
        return - state * state * state * state * state * state * state * state.gradient(bc=self.bc)[
            0] + self.nu * state.laplace(bc=self.bc)


class ChafeeInfantePDE(PDEBase):

    def __init__(self, nu=1, bc='natural'):
        """ initialize the class with a diffusivity and boundary conditions
        for the actual field and its second derivative """
        self.nu = nu
        self.bc = bc

    def evolution_rate(self, state, t=0):
        """ numpy implementation of the evolution equation """
        return - self.nu * (state * state * state - state) + state.laplace(bc=self.bc)


# pde solver  
def solve_PDE_from_initial_conditions(grid_size, data, nu, final_t, dt, pde_class):
    grid = CartesianGrid([[0, 2 * np.pi]], [grid_size], periodic=True)
    state = ScalarField(grid, data=data)
    # solve the equation and store the trajectory
    storage = MemoryStorage()
    eq = pde_class(nu=nu)
    solver = pde.ScipySolver(eq)
    controller = pde.Controller(solver,
                                t_range=final_t,
                                tracker=storage.tracker(dt))

    sol2 = controller.run(state)

    return storage.data


# utils
def get_variables(num, loader):
    with torch.no_grad():
        for i, (x, y, t) in enumerate(loader):
            if i == num:
                x, y, t = x.cuda(), y.cuda(), t.cuda()
                break
    return x, y, t


def to_numpy(x):
    return x.cpu().detach().numpy()


def get_old_data(sub, s, ntrain, ntest, path):
    dataloader = MatReader(path)
    x_data = dataloader.read_field('a')[:, ::sub]
    y_data = dataloader.read_field('u')[:, ::sub]
    x_train = x_data[:ntrain, :]
    y_train = y_data[:ntrain, :]
    x_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :]
    # cat the locations information
    grid = np.linspace(0, 2 * np.pi, s).reshape(1, s, 1)
    grid = torch.tensor(grid, dtype=torch.float)
    x_train = torch.cat(
        [x_train.reshape(ntrain, s, 1),
         grid.repeat(ntrain, 1, 1)], dim=2)
    x_test = torch.cat([x_test.reshape(ntest, s, 1),
                        grid.repeat(ntest, 1, 1)],
                       dim=2)
    t_train = torch.tensor([[1.0]] * x_train.shape[0])
    t_test = torch.tensor([[1.0]] * x_test.shape[0])

    return torch.utils.data.TensorDataset(
        x_train, y_train,
        t_train), torch.utils.data.TensorDataset(x_test, y_test, t_test)


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, help='Path to burgers_data_R10.mat',
                    default='burgers_data_R10.mat')
parser.add_argument('-ntr', '--num_train', type=int, help='Number of training samples', default=1000)
parser.add_argument('-nts', '--num_test', type=int, help='Number of testing samples', default=100)
parser.add_argument('-o', '--output', type=str, help='Output path', default='data')
parser.add_argument('-pde', '--pde', type=str, help='PDE type', default='gburgers2', choices=['gburgers2',
                                                                                              'gburgers3',
                                                                                              'gburgers4',
                                                                                              'chafee'])

if __name__ == '__main__':
    args = vars(parser.parse_args())
    datapath = args['data_path']
    ntrain = args['num_train']
    ntest = args['num_test']
    outpath = args['output']
    pde_name = args['pde']
    os.makedirs(outpath, exist_ok=True)
    eq_dict = {
        'gburgers2': GeneralizedBurgers2PDE,
        'gburgers3': GeneralizedBurgers3PDE,
        'gburgers4': GeneralizedBurgers4PDE,
        'chafee': ChafeeInfantePDE}

    file_dict = {'gburgers2': 'burgers_gen_2',
                 'gburgers3': 'burgers_gen_3',
                 'gburgers4': 'burgers_gen_4',
                 'chafee': 'ChafeeInfantePDE'}

    sub = 2 ** 3  # subsampling rate
    s = 2 ** 13 // sub  # total grid size divided by the subsampling rate

    old_train_data, old_test_data = get_old_data(sub, s, ntrain, ntest, datapath)
    final_train_loader = torch.utils.data.DataLoader(old_train_data, batch_size=1, shuffle=False)
    final_test_loader = torch.utils.data.DataLoader(old_test_data, batch_size=1, shuffle=False)

    y_pde_dict = {}
    for k in tqdm(range(ntrain)):
        try:
            x, y, t = get_variables(k, final_train_loader)
            grid = x[:, :, 1].cuda()
            v0_net = x[:, :, 0].cuda()
            final_t = 1

            y_pde = solve_PDE_from_initial_conditions(len(to_numpy(v0_net[0])),
                                                      to_numpy(v0_net[0]),
                                                      nu=0.1,
                                                      final_t=final_t,
                                                      dt=1, pde_class=eq_dict[pde_name])
            y_pde_dict[k] = y_pde
        except:
            print('failed to compute for k = ', k)
    np.save(os.path.join(outpath, f'{file_dict[args["pde"]]}_train.npy'), y_pde_dict)

    y_pde_dict = {}
    for k in tqdm(range(ntest)):
        try:
            x, y, t = get_variables(k, final_test_loader)
            grid = x[:, :, 1].cuda()
            v0_net = x[:, :, 0].cuda()
            final_t = 3

            y_pde = solve_PDE_from_initial_conditions(len(to_numpy(v0_net[0])),
                                                      to_numpy(v0_net[0]),
                                                      nu=0.1,
                                                      final_t=final_t,
                                                      dt=0.01, pde_class=eq_dict[pde_name])
            y_pde_dict[k] = y_pde
        except:
            print('failed to compute for k = ', k)
    np.save(os.path.join(outpath, f'{file_dict[args["pde"]]}_test.npy'), y_pde_dict)
