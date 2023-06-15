import argparse
import os


def parser_1d():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--circular', action='store_true', help='Circular convolutions')
    parser.add_argument('-d', '--datasource', type=str, default='newold')
    parser.add_argument('--use_tanh', action='store_true', help='use tanh normalization on time')
    parser.add_argument('--weighted', action='store_true', help='divide loss by the number of orders')
    parser.add_argument('--nshot', type=int, help='Number of samples to use during training', default=-1)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=500)
    parser.add_argument('--order', type=int, help='Order of the loss composition', default=1)
    parser.add_argument('-g', '--genorder', type=int, help='Order of generalized burgers eq', default=2)
    parser.add_argument('--uniformintervals', action='store_true', help='split composition time in uniform intervals')
    parser.add_argument('--vanilla', action='store_true', help='Use the original FNO')
    parser.add_argument('--sub', type=int, help='Subsampling', default=2 ** 3)
    parser.add_argument('-i', '--inter', action='store_true', help='Use intermediate term in the loss function')
    parser.add_argument('-r', '--randomshift', action='store_true', help='Random Shift of the input')
    parser.add_argument('-s', '--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--use_gelu', action='store_true', help='use GeLU activation')
    parser.add_argument('--datapath', type=str)

    return parser


def parser_2d():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--circular', action='store_true', help='Circular convolutions')
    parser.add_argument('-d', '--datasource', type=str, default='navier')
    parser.add_argument('--use_tanh', action='store_true', help='use tanh normalization on time')
    parser.add_argument('--weighted', action='store_true', help='divide loss by the number of orders')
    parser.add_argument('--nshot', type=int, help='Number of samples to use during training', default=-1)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=500)
    parser.add_argument('--order', type=int, help='Order of the loss composition', default=1)
    parser.add_argument('--uniformintervals', action='store_true', help='split composition time in uniform intervals')
    parser.add_argument('--vanilla', action='store_true', help='Use the original FNO')
    parser.add_argument('-i', '--inter', action='store_true', help='Use intermediate term in the loss function')
    parser.add_argument('-s', '--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--use_gelu', action='store_true', help='use GeLU activation')

    return parser


def parser_3d():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--circular', action='store_true', help='Circular convolutions')
    parser.add_argument('-d', '--datasource', type=str, default='navier')
    parser.add_argument('--use_tanh', action='store_true', help='use tanh normalization on time')
    parser.add_argument('--weighted', action='store_true', help='divide loss by the number of orders')
    parser.add_argument('--nshot', type=int, help='Number of samples to use during training', default=-1)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=500)
    parser.add_argument('--order', type=int, help='Order of the loss composition', default=1)
    parser.add_argument('--uniformintervals', action='store_true', help='split composition time in uniform intervals')
    parser.add_argument('--vanilla', action='store_true', help='Use the original FNO')
    parser.add_argument('-i', '--inter', action='store_true', help='Use intermediate term in the loss function')
    parser.add_argument('-s', '--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--use_gelu', action='store_true', help='use GeLU activation')

    return parser


def gen_save_path_1d(args):
    nshot_str = f'nshot-{args["nshot"]}' if args['nshot'] != -1 else ""
    randomshift_str = 'randomshift' if args['randomshift'] else ''
    inter_str = 'inter' if args['inter'] else ''
    weighted_str = 'weighted' if args['weighted'] else ''
    uniformintervals_str = 'uniformintervals' if args['uniformintervals'] else ''
    exp_name = 'FNO' if args['vanilla'] else 'hyper'
    if args['datasource'] == 'old':
        foldername = 'burgers'
    elif args['datasource'] == 'genburgers':
        foldername = args['datasource'] + str(args['genorder'])
    else:
        foldername = args["datasource"]
    save_path = os.path.join('checkpoints', '1D',
                             f'{foldername}{exp_name}{args["datasource"]}{"_tanh" if args["use_tanh"] else ""}{nshot_str}order_{args["order"]}{randomshift_str}{inter_str}{weighted_str}{uniformintervals_str}sub-{8192 // args["sub"]}{"_gelu" if args["use_gelu"] else ""}')

    return save_path


def gen_save_path_2d(args):
    nshot_str = f'nshot-{args["nshot"]}' if args['nshot'] != -1 else ""
    inter_str = 'inter' if args['inter'] else ''
    weighted_str = 'weighted' if args['weighted'] else ''
    uniformintervals_str = 'uniformintervals' if args['uniformintervals'] else ''
    exp_name = 'FNO' if args['vanilla'] else 'hyper'
    if args['datasource'] == 'old':
        foldername = 'darcy'
    else:
        foldername = args["datasource"]
    save_path = os.path.join('checkpoints', '2D',
                             f'{foldername}{exp_name}{args["datasource"]}{"_tanh" if args["use_tanh"] else ""}{nshot_str}order_{args["order"]}{inter_str}{weighted_str}{uniformintervals_str}{"_gelu" if args["use_gelu"] else ""}')

    return save_path


def gen_save_path_3d(args):
    nshot_str = f'nshot-{args["nshot"]}' if args['nshot'] != -1 else ""
    inter_str = 'inter' if args['inter'] else ''
    weighted_str = 'weighted' if args['weighted'] else ''
    uniformintervals_str = 'uniformintervals' if args['uniformintervals'] else ''
    exp_name = 'FNO' if args['vanilla'] else 'hyper'
    if args['datasource'] == 'old':
        foldername = 'darcy'
    else:
        foldername = args["datasource"]
    save_path = os.path.join('checkpoints', '3D',
                             f'{foldername}{exp_name}{args["datasource"]}{"_tanh" if args["use_tanh"] else ""}{nshot_str}order_{args["order"]}{inter_str}{weighted_str}{uniformintervals_str}{"_gelu" if args["use_gelu"] else ""}')

    return save_path
