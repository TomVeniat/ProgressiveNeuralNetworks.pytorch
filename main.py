import argparse

import torch
from torch.autograd import Variable
from tqdm import tqdm

from src.data.PermutedMNIST import get_permuted_MNIST
from src.model.ProgressiveNeuralNetworks import PNN
from src.tools.arg_parser_actions import LengthCheckAction


def get_args():
    parser = argparse.ArgumentParser(description='Progressive Neural Networks')
    parser.add_argument('-path', default='/local/veniat/data', type=str, help='path to the data')

    parser.add_argument('--layers', metavar='L', type=int, default=4, help='Number of layers per task')
    parser.add_argument('--sizes', dest='sizes', default=[8, 16, 256, 128], nargs='+', action=LengthCheckAction)
    parser.add_argument('--bs', dest='batch_size', type=int, default=3)

    args = parser.parse_known_args()
    return args[0]


def main(args):

    data_train, data_val, data_test = get_permuted_MNIST(args['path'], args['batch_size'])

    for batch in tqdm(data_train):
        # print(batch)
        pass


    model = PNN(args['layers'], args['sizes'][0], args['sizes'][1])
    # model = PNN(args['layers'], args['sizes'])
    model.new_task()
    model.new_task()
    model.freeze_columns()
    model.new_task()
    t = torch.Tensor(args['batch_size'], args['sizes'][0])

    model(Variable(t))
    print(model)


if __name__ == '__main__':
    main(vars(get_args()))
