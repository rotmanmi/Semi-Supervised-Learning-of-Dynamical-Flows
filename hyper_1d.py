import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from hypernet.hypernet1d import HyperSimpleBlock1d, SimpleBlock1d, HyperSimpleBlock1dGELU, SimpleBlock1dGELU
from functools import reduce
import json
from timeit import default_timer

import os

from utilities3 import MatReader, save_checkpoint, LpLoss
from utils.parsers import parser_1d, gen_save_path_1d

torch.manual_seed(0)
np.random.seed(0)

parser = parser_1d()


class Net1d(nn.Module):
    def __init__(self, modes, width, args):
        super(Net1d, self).__init__()

        """
        A wrapper function
        """
        self.use_tanh = args['use_tanh']
        self.vanilla = args['vanilla']
        if args['vanilla']:
            if args['use_gelu']:
                self.conv1 = SimpleBlock1dGELU(modes, width, args)
            else:
                self.conv1 = SimpleBlock1d(modes, width, args)
        else:
            if args['use_gelu']:
                self.conv1 = HyperSimpleBlock1dGELU(modes, width)
            else:
                self.conv1 = HyperSimpleBlock1d(modes, width)

    def forward(self, x, t):
        if self.use_tanh:
            t = torch.tanh(t)
        else:
            t = t / 10.0
        if self.vanilla:
            x = self.conv1(x)
        else:
            x = self.conv1(x, t)
        return x.squeeze(-1)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


################################################################
#  configurations
################################################################
ntrain = 1000
ntest = 100

# sub = 2 ** 3  # subsampling rate


batch_size = 20
learning_rate = 0.001

step_size = 100
gamma = 0.5

modes = 16
width = 64


def get_old_data(datapath, sub):
    h = 2 ** 13 // sub  # total grid size divided by the subsampling rate
    s = h
    dataloader = MatReader(os.path.join(datapath, 'burgers_data_R10.mat'))
    x_data = dataloader.read_field('a')[:, ::sub]
    y_data = dataloader.read_field('u')[:, ::sub]
    x_train = x_data[:ntrain, :]
    y_train = y_data[:ntrain, :]
    x_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :]
    # cat the locations information
    grid = np.linspace(0, 2 * np.pi, s).reshape(1, s, 1)
    grid = torch.tensor(grid, dtype=torch.float)
    x_train = torch.cat([x_train.reshape(ntrain, s, 1), grid.repeat(ntrain, 1, 1)], dim=2)
    x_test = torch.cat([x_test.reshape(ntest, s, 1), grid.repeat(ntest, 1, 1)], dim=2)
    t_train = torch.tensor([[1.0]] * x_train.shape[0])
    t_test = torch.tensor([[1.0]] * x_test.shape[0])

    return torch.utils.data.TensorDataset(x_train, y_train, t_train), torch.utils.data.TensorDataset(x_test, y_test,
                                                                                                     t_test)


def get_Chafee_data(datapath, sub):
    h = 2 ** 13 // 8  # total grid size divided by the subsampling rate
    s = h
    dataloader_train = np.load(os.path.join(datapath, 'y_pde_dict_nu01t_0_1_ChafeeInfante_train.npy'),
                               allow_pickle=True).item()
    dataloader_test = np.load(os.path.join(datapath, 'y_pde_dict_nu01t_0_3_ChafeeInfante_test.npy'),
                              allow_pickle=True).item()

    x_data = torch.from_numpy(
        np.stack([x[0] for _, x in dataloader_train.items()] + [x[0] for _, x in dataloader_test.items()], 0).astype(
            np.float32))
    y_data = torch.from_numpy(
        np.stack([x[1] for _, x in dataloader_train.items()] + [x[100] for _, x in dataloader_test.items()], 0).astype(
            np.float32))
    x_train = x_data[:ntrain, :]
    y_train = y_data[:ntrain, :]
    x_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :]

    # cat the locations information
    grid = np.linspace(0, 2 * np.pi, s).reshape(1, s, 1)
    grid = torch.tensor(grid, dtype=torch.float)
    x_train = torch.cat([x_train.reshape(ntrain, s, 1), grid.repeat(ntrain, 1, 1)], dim=2)
    x_test = torch.cat([x_test.reshape(ntest, s, 1), grid.repeat(ntest, 1, 1)], dim=2)
    t_train = torch.tensor([[1.0]] * x_train.shape[0])
    t_test = torch.tensor([[1.0]] * x_test.shape[0])

    # Duplicating t=0 as identity:
    # t_train = torch.tensor([[1.0]] * x_train.shape[0] + [[0.0]] * x_train.shape[0])
    # x_train = torch.cat([x_train, x_train], dim=0)
    # y_train = torch.tensor(np.concatenate([y_train, x_data[:ntrain, :]], axis=0))

    return torch.utils.data.TensorDataset(x_train, y_train, t_train), torch.utils.data.TensorDataset(x_test, y_test,
                                                                                                     t_test)


def get_genburgers_data(datapath, sub, gen_burgers_order):
    h = 2 ** 13 // 8  # total grid size divided by the subsampling rate
    s = h
    dataloader_train = np.load(os.path.join(datapath, f'burgers_gen_{gen_burgers_order}_train.npy'),
                               allow_pickle=True).item()
    dataloader_test = np.load(os.path.join(datapath, f'burgers_gen_{gen_burgers_order}_test.npy'),
                              allow_pickle=True).item()

    x_data = torch.from_numpy(
        np.stack([x[0] for _, x in dataloader_train.items()] + [x[0] for _, x in dataloader_test.items()], 0).astype(
            np.float32))
    y_data = torch.from_numpy(
        np.stack([x[1] for _, x in dataloader_train.items()] + [x[100] for _, x in dataloader_test.items()], 0).astype(
            np.float32))
    x_train = x_data[:ntrain, :]
    y_train = y_data[:ntrain, :]
    x_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :]

    # cat the locations information
    grid = np.linspace(0, 2 * np.pi, s).reshape(1, s, 1)
    grid = torch.tensor(grid, dtype=torch.float)
    x_train = torch.cat([x_train.reshape(ntrain, s, 1), grid.repeat(ntrain, 1, 1)], dim=2)
    x_test = torch.cat([x_test.reshape(ntest, s, 1), grid.repeat(ntest, 1, 1)], dim=2)
    t_train = torch.tensor([[1.0]] * x_train.shape[0])
    t_test = torch.tensor([[1.0]] * x_test.shape[0])

    return torch.utils.data.TensorDataset(x_train, y_train, t_train), torch.utils.data.TensorDataset(x_test, y_test,
                                                                                                     t_test)


def higher_order_loss_composition(model, x, y, t, lossfunc, order):
    prev_t = torch.zeros_like(t)
    cur_x = x[..., 0]

    if order > 0:
        t_samples = torch.stack([torch.distributions.uniform.Uniform(prev_t, t).sample() for _ in range(order)], -1)
        t_samples, _ = torch.sort(t_samples, -1)

    for i in range(order):
        cur_t = t_samples[..., i]
        cur_x = model(torch.stack([cur_x, x[..., 1]], -1), cur_t - prev_t)
        prev_t = cur_t

    cur_x = model(torch.stack([cur_x, x[..., 1]], -1), t - prev_t)
    loss = lossfunc(cur_x.view(x.shape[0], -1), y.view(x.shape[0], -1))

    return loss


def higher_order_loss_composition_intervals(model, x, y, t, lossfunc, order):
    prev_t = torch.zeros_like(t)
    cur_x = x[..., 0]

    if order > 0:
        t_samples = torch.stack(
            [torch.distributions.uniform.Uniform((o / order) * t, ((o + 1) * t) / order).sample() for o in
             range(order)], -1)

    for i in range(order):
        cur_t = t_samples[..., i]
        cur_x = model(torch.stack([cur_x, x[..., 1]], -1), cur_t - prev_t)
        prev_t = cur_t

    cur_x = model(torch.stack([cur_x, x[..., 1]], -1), t - prev_t)
    loss = lossfunc(cur_x.view(x.shape[0], -1), y.view(x.shape[0], -1))

    return loss


def intermediate_term(model, x, t, lossfunc):
    prev_t = torch.zeros_like(t)

    t_samples = torch.stack([torch.distributions.uniform.Uniform(prev_t, t).sample() for _ in range(2)], -1)
    t_samples, _ = torch.sort(t_samples, -1)

    cur_x1 = model(x, t_samples[..., 0])
    cur_x2 = model(torch.stack([cur_x1, x[..., 1]], -1), (t_samples[..., 1] - t_samples[..., 0]))
    cur_x12 = model(x, t_samples[..., 1])
    loss = lossfunc(cur_x2.view(x.shape[0], -1), cur_x12.view(x.shape[0], -1))

    return loss


def intermediate_term_commutative(model, x, t, lossfunc):
    prev_t = torch.zeros_like(t)

    t_samples = torch.stack([torch.distributions.uniform.Uniform(prev_t, t).sample() for _ in range(2)], -1)
    t_samples, _ = torch.sort(t_samples, -1)

    t1 = t_samples[..., 0]
    t2 = t_samples[..., 1] - t1

    cur_x1 = model(x, t1)
    cur_x2 = model(x, t2)
    cur_x12 = model(torch.stack([cur_x1, x[..., 1]], -1), t2)
    cur_x21 = model(torch.stack([cur_x2, x[..., 1]], -1), t1)
    loss = lossfunc(cur_x12.view(x.shape[0], -1), cur_x21.view(x.shape[0], -1))

    return loss


################################################################
# read data
################################################################
if __name__ == '__main__':
    args = vars(parser.parse_args())
    print(args)
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    epochs = args['epochs']
    sub = args['sub']
    h = 2 ** 13 // sub  # total grid size divided by the subsampling rate
    s = h
    # Data is of the shape (number of samples, grid size)
    if args['datasource'] == 'chafee':
        old_train_data, old_test_data = get_Chafee_data(args['datapath'], sub=1)
        args['sub'] = 8
        train_dataset = old_train_data
        test_dataset = old_test_data
        final_test_dataset = old_test_data

    elif args['datasource'] == 'genburgers':
        old_train_data, old_test_data = get_genburgers_data(args['datapath'], sub=1,
                                                            gen_burgers_order=args['genorder'])
        args['sub'] = 8
        train_dataset = old_train_data
        test_dataset = old_test_data
        final_test_dataset = old_test_data
    else:
        old_train_data, old_test_data = get_old_data(args['datapath'], sub)

        train_dataset = old_train_data
        test_dataset = old_test_data
        final_test_dataset = old_test_data

    if args['nshot'] != -1:
        train_dataset = torch.utils.data.Subset(train_dataset, np.arange(0, args['nshot']))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    final_test_loader = torch.utils.data.DataLoader(final_test_dataset, batch_size=1, shuffle=False)

    # model
    model = Net1d(modes, width, args).cuda()
    print(model.count_params())

    ################################################################
    # training and evaluation
    ################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    if args['nshot'] != -1:
        step_size = step_size / (args['nshot'] / float(ntrain))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0

        for x, y, t in train_loader:

            if args['randomshift']:
                shift_params = torch.randint(x.shape[1], (x.shape[0],))
                shifted_x = torch.stack([torch.roll(xt, sp.item()) for xt, sp in zip(x[..., 0], shift_params)], 0)
                shifted_y = torch.stack([torch.roll(xt, sp.item()) for xt, sp in zip(y, shift_params)], 0)

                total_shifted_x = torch.stack([shifted_x, x[..., 1]], -1)
                x = total_shifted_x.detach()
                y = shifted_y.detach()

            x, y = x.cuda(), y.cuda()
            t = t.cuda()
            optimizer.zero_grad()
            out = model(x, t)
            mse = F.mse_loss(out, y, reduction='mean')

            if args['vanilla']:
                l = myloss(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
            else:
                out_initial = model(x, torch.zeros_like(t))

                l2_inter = 0.0
                for ord in range(args['order'] + 1):
                    if args['uniformintervals']:
                        l2_inter += higher_order_loss_composition_intervals(model, x, y, t, myloss, ord)
                    else:
                        l2_inter += higher_order_loss_composition(model, x, y, t, myloss, ord)

                if args['weighted']:
                    l2_inter /= (args['order'] + 1)
                l2_initial = myloss(out_initial.view(x.shape[0], -1), x[..., 0].view(x.shape[0], -1))
                l_inter = 0
                if args['inter']:
                    l_inter = intermediate_term(model, x, t, myloss)
                l = l2_initial + l2_inter + l_inter
            l.backward()  # use the l2 relative loss

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l.item()

        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y, t in test_loader:
                x, y, t = x.cuda(), y.cuda(), t.cuda()

                out = model(x, t)
                test_l2 += myloss(out.view(x.shape[0], -1), y.view(x.shape[0], -1)).item()

        train_mse /= len(train_dataset)
        train_l2 /= len(train_dataset)
        test_l2 /= len(test_dataset)

        t2 = default_timer()
        print(ep, t2 - t1, train_mse, train_l2, test_l2)

    exp_name = 'FNO' if args['vanilla'] else 'hyper'
    save_path = gen_save_path_1d(args)
    os.makedirs(save_path, exist_ok=True)
    save_checkpoint(os.path.join(save_path, 'model_{:08d}'.format(ep)),
                    {'model_state_dict': model.state_dict(), 'epoch': ep})
    index = 0

    test_l2 = 0
    with torch.no_grad():
        for x, y, t in final_test_loader:
            x, y, t = x.cuda(), y.cuda(), t.cuda()
            out = model(x, t)

            test_l2 += myloss(out.view(x.shape[0], -1), y.view(x.shape[0], -1)).item()
            print(index, test_l2)
            index = index + 1

    print(test_l2 / float(index))
    np.savetxt(os.path.join(save_path, f'results_{args["seed"]}.txt'), [[float(test_l2 / float(index))]])
    res_dict = {'l2': float(test_l2 / float(index)),
                'exp_name': exp_name,
                'nshot': args['nshot'],
                'randomshift': args['randomshift'],
                'dataset': args['datasource'],
                'order': args['order'],
                'epochs': args['epochs'],
                'tanh': args['use_tanh'],
                'weighted': args['weighted'],
                'intermediate': args['inter'],
                'uniformintervals': args['uniformintervals'],
                'sub': args['sub'],
                'gelu': args['use_gelu']}

    with open(os.path.join(save_path, f'results_{args["seed"]}.json'), 'w') as f:
        json.dump(res_dict, f)
