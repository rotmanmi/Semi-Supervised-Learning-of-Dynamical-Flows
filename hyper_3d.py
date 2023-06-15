import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import json
import operator
from hypernet.hypernet3d import HyperSimpleBlock3d, SimpleBlock3d, SimpleBlock3dGeLU, HyperSimpleBlock3dGeLU

from functools import reduce

from timeit import default_timer
from utilities3 import *
from utils.parsers import parser_3d, gen_save_path_3d

torch.manual_seed(0)
np.random.seed(0)

parser = parser_3d()


class Net3d(nn.Module):
    def __init__(self, modes, width, args, **kwargs):
        super().__init__()

        """
        A wrapper function
        """
        self.use_tanh = args['use_tanh']
        self.vanilla = args['vanilla']

        if args['vanilla']:
            if args['use_gelu']:
                self.conv1 = SimpleBlock3dGeLU(modes, modes, modes, width, args, **kwargs)
            else:
                self.conv1 = SimpleBlock3d(modes, modes, modes, width, args, **kwargs)
        else:
            if args['use_gelu']:
                self.conv1 = HyperSimpleBlock3dGeLU(modes, modes, modes, width, **kwargs)
            else:
                self.conv1 = HyperSimpleBlock3d(modes, modes, modes, width, **kwargs)

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


def higher_order_loss_composition_3d(model, x, y, t, lossfunc, order, y_normalizer):
    prev_t = torch.zeros_like(t)
    cur_x = x[..., :3]

    if order > 0:
        t_samples = torch.stack([torch.distributions.uniform.Uniform(prev_t, t).sample() for _ in range(order)], -1)
        t_samples, _ = torch.sort(t_samples, -1)

    for i in range(order):
        cur_t = t_samples[..., i]
        cur_x = model(torch.cat([cur_x, x[..., 3:]], -1), cur_t - prev_t)
        prev_t = cur_t

    cur_x = model(torch.cat([cur_x, x[..., 3:]], -1), t - prev_t)
    cur_x = y_normalizer.decode(cur_x)
    loss = lossfunc(cur_x.view(x.shape[0], -1), y.view(x.shape[0], -1))

    return loss


def higher_order_loss_composition_intervals_3d(model, x, y, t, lossfunc, order, y_normalizer):
    prev_t = torch.zeros_like(t)
    cur_x = x[..., :3]

    if order > 0:
        t_samples = torch.stack(
            [torch.distributions.uniform.Uniform((o / order) * t, ((o + 1) * t) / order).sample() for o in
             range(order)], -1)

    for i in range(order):
        cur_t = t_samples[..., i]
        cur_x = model(torch.cat([cur_x, x[..., 3:]], -1), cur_t - prev_t)
        prev_t = cur_t

    cur_x = model(torch.cat([cur_x, x[..., 3:]], -1), t - prev_t)
    cur_x = y_normalizer.decode(cur_x)
    loss = lossfunc(cur_x.view(x.shape[0], -1), y.view(x.shape[0], -1))

    return loss


def intermediate_term_3d(model, x, t, lossfunc, y_normalizer):
    prev_t = torch.zeros_like(t)
    cur_x = x[..., :3]

    t_samples = torch.stack([torch.distributions.uniform.Uniform(prev_t, t).sample() for _ in range(2)], -1)
    t_samples, _ = torch.sort(t_samples, -1)

    cur_x1 = model(torch.cat([cur_x, x[..., 3:]], -1), t_samples[..., 0])
    cur_x2 = y_normalizer.decode(model(torch.cat([cur_x1, x[..., 3:]], -1), (t_samples[..., 1] - t_samples[..., 0])))
    cur_x12 = y_normalizer.decode(model(torch.cat([cur_x, x[..., 3:]], -1), t_samples[..., 1]))
    loss = lossfunc(cur_x2.view(x.shape[0], -1), cur_x12.view(x.shape[0], -1))

    return loss


def get_3dnavier_data():
    TRAIN_PATH = '/mntssd/michael/data/navier3d.npz'
    loader = np.load(TRAIN_PATH)
    x = np.stack([loader['initial_v_x'], loader['initial_v_y'], loader['initial_v_z']], -1).astype(np.float32)
    y = np.stack([loader['final_v_x'], loader['final_v_y'], loader['final_v_z']], -1).astype(np.float32)

    x_train = x[:ntrain]
    x_test = x[-ntest:]
    y_train = y[:ntrain]
    y_test = y[-ntest:]

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)
    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

    grids = []
    grids.append(np.linspace(0, 1, x_train.shape[-2]))
    grids.append(np.linspace(0, 1, x_train.shape[-3]))
    grids.append(np.linspace(0, 1, x_train.shape[-4]))

    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    grid = grid.reshape(1, x_train.shape[-4], x_train.shape[-3], x_train.shape[-2], 3)
    grid = torch.tensor(grid, dtype=torch.float)
    x_train = torch.cat(
        [x_train.reshape(ntrain, x_train.shape[-4], x_train.shape[-3], x_train.shape[-2], x_train.shape[-1]),
         grid.repeat(ntrain, 1, 1, 1, 1)], dim=4)
    x_test = torch.cat(
        [x_test.reshape(ntest, x_test.shape[-4], x_test.shape[-3], x_test.shape[-2], x_test.shape[-1]),
         grid.repeat(ntest, 1, 1, 1, 1)],
        dim=4)

    t_train = torch.tensor([[1.0]] * x_train.shape[0])
    t_test = torch.tensor([[1.0]] * x_test.shape[0])

    return torch.utils.data.TensorDataset(x_train, y_train, t_train), torch.utils.data.TensorDataset(x_test, y_test,
                                                                                                     t_test), y_normalizer


if __name__ == '__main__':

    ################################################################
    # configs
    ################################################################

    ntrain = 1000
    ntest = 100

    batch_size = 10
    learning_rate = 0.001

    epochs = 500
    step_size = 100
    gamma = 0.5

    modes = 4
    width = 20

    r = 5
    h = int(((421 - 1) / r) + 1)
    s = h

    ################################################################
    # load data and data normalization
    ################################################################

    args = vars(parser.parse_args())
    print(args)
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    epochs = args['epochs']
    if args['datasource'] == 'navier':
        train_dataset, test_dataset, y_normalizer = get_3dnavier_data()
        in_channels = 6
    else:
        exit('No such dataset')
    final_test_dataset = test_dataset

    if args['nshot'] != -1:
        train_dataset = torch.utils.data.Subset(train_dataset, np.arange(0, args['nshot']))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    final_test_loader = torch.utils.data.DataLoader(final_test_dataset, batch_size=1, shuffle=False)

    ################################################################
    # training and evaluation
    ################################################################
    model = Net3d(modes, width, args, in_channels=in_channels).cuda()
    print(model.count_params())

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = LpLoss(size_average=False)
    y_normalizer.cuda()

    exp_name = 'FNO' if args['vanilla'] else 'hyper'
    save_path = gen_save_path_3d(args)
    os.makedirs(save_path, exist_ok=True)

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        for x, y, t in train_loader:
            x, y, t = x.cuda(), y.cuda(), t.cuda()

            optimizer.zero_grad()
            out = model(x, t)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

            if args['vanilla']:
                l = myloss(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
            else:
                out_initial = y_normalizer.decode(model(x, torch.zeros_like(t)))
                l2_initial = myloss(out_initial.view(x.shape[0], -1),
                                    y_normalizer.decode(x[..., :(in_channels - 3)]).view(x.shape[0], -1))

                l2_inter = 0.0
                for ord in range(args['order'] + 1):
                    if args['uniformintervals']:
                        l2_inter += higher_order_loss_composition_intervals_3d(model, x, y, t, myloss, ord,
                                                                               y_normalizer)
                    else:
                        l2_inter += higher_order_loss_composition_3d(model, x, y, t, myloss, ord, y_normalizer)
                if args['weighted']:
                    l2_inter /= (args['order'] + 1)

                l_inter = 0.0
                if args['inter']:
                    l_inter += intermediate_term_3d(model, x, t, myloss, y_normalizer)
                # l_temp = myloss(out.view(x.shape[0], -1), y.view(x.shape[0], -1))

                l = l2_initial + l2_inter + 0.1 * l_inter
            l.backward()

            optimizer.step()
            train_mse += l.item()

        scheduler.step()

        model.eval()
        abs_err = 0.0
        rel_err = 0.0
        with torch.no_grad():
            for x, y, t in test_loader:
                x, y, t = x.cuda(), y.cuda(), t.cuda()

                # out = model(x, t)
                out = y_normalizer.decode(model(x, t))

                rel_err += myloss(out.view(x.shape[0], -1), y.view(x.shape[0], -1)).item()

        train_mse /= ntrain
        abs_err /= ntest
        rel_err /= ntest

        t2 = default_timer()
        print(ep, t2 - t1, train_mse, rel_err, flush=True)
        if ep % 100 == 0 and ep > 0:
            save_checkpoint(os.path.join(save_path, 'model_{:08d}'.format(ep)),
                            {'model_state_dict': model.state_dict(), 'epoch': ep})

    save_checkpoint(os.path.join(save_path, 'model_{:08d}'.format(ep)),
                    {'model_state_dict': model.state_dict(), 'epoch': ep})
    index = 0
    model.eval()
    rel_err = 0.0
    with torch.no_grad():
        for x, y, t in final_test_loader:
            x, y, t = x.cuda(), y.cuda(), t.cuda()

            # out = model(x, t)
            out = y_normalizer.decode(model(x, t))

            rel_err += myloss(out.view(1, -1), y.view(1, -1)).item()
            index = index + 1
            print(index, rel_err / float(index))

    print(rel_err / float(index))
    np.savetxt(os.path.join(save_path, f'results_{args["seed"]}.txt'), [[float(rel_err / float(index))]])
    res_dict = {'l2': float(rel_err / float(index)),
                'exp_name': exp_name,
                'nshot': args['nshot'],
                'dataset': args['datasource'],
                'order': args['order'],
                'epochs': args['epochs'],
                'tanh': args['use_tanh'],
                'weighted': args['weighted'],
                'intermediate': args['inter'],
                'uniformintervals': args['uniformintervals'],
                'gelu': args['use_gelu']
                }

    with open(os.path.join(save_path, f'results_{args["seed"]}.json'), 'w') as f:
        json.dump(res_dict, f)
