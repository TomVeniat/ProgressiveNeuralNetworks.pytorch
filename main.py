import argparse

import os
import torch
import torch.nn.functional as F

import logging

from torch.autograd import Variable
from tqdm import tqdm

from src.data.PermutedMNIST import get_permuted_MNIST
from src.model.ProgressiveNeuralNetworks import PNN
from src.tools.arg_parser_actions import LengthCheckAction
from src.tools.evaluation import evaluate_model

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser(description='Progressive Neural Networks')
    parser.add_argument('-path', default='/local/veniat/data', type=str, help='path to the data')
    parser.add_argument('-cuda', default=-1, type=int, help='Cuda device to use (-1 for none)')

    parser.add_argument('--layers', metavar='L', type=int, default=4, help='Number of layers per task')
    parser.add_argument('--sizes', dest='sizes', default=[784, 256, 256, 256, 10], nargs='+',
                        action=LengthCheckAction)

    parser.add_argument('--n_tasks', dest='n_tasks', type=int, default=5)
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--bs', dest='batch_size', type=int, default=50)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='Optimizer learning rate')
    parser.add_argument('--wd', dest='wd', type=float, default=1e-4, help='Optimizer weight decay')
    parser.add_argument('--momentum', dest='momentum', type=float, default=1e-4, help='Optimizer momentum')

    args = parser.parse_known_args()
    return args[0]


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['cuda'])

    model = PNN(args['layers'])

    tasks_data = [get_permuted_MNIST(args['path'], args['batch_size']) for _ in range(args['n_tasks'])]

    x = torch.Tensor()
    y = torch.LongTensor()

    if args['cuda'] != -1:
        logger.info('Running with cuda (GPU n°{})'.format(args['cuda']))
        model.cuda()
        x = x.cuda()
        y = y.cuda()
    else:
        logger.warning('Running WITHOUT cuda')

    for task_id, (train_set, val_set, test_set) in enumerate(tasks_data):
        # val_perf = evaluate_model(model, x, y, val_set, task_id=task_id)

        model.freeze_columns()
        model.new_task(args['sizes'])

        optimizer = torch.optim.RMSprop(model.parameters(task_id), lr=args['lr'],
                                        weight_decay=args['wd'], momentum=args['momentum'])
        for epoch in range(args['epochs']):
            total_samples = 0
            total_loss = 0
            correct_samples = 0
            for inputs, labels in tqdm(train_set):
                x.resize_(inputs.size()).copy_(inputs)
                y.resize_(labels.size()).copy_(labels)

                x = x.view(x.size(0), -1)
                predictions = model(Variable(x))

                _, predicted = torch.max(predictions.data, 1)
                total_samples += y.size(0)
                correct_samples += (predicted == y).sum()

                indiv_loss = F.cross_entropy(predictions, Variable(y))
                total_loss += indiv_loss.data[0]

                optimizer.zero_grad()
                indiv_loss.backward()
                optimizer.step()

            logger.info(
                '[T{}][{}/{}] Loss={}, Acc= {}'.format(task_id, epoch, args['epochs'], total_loss / total_samples,
                                                       correct_samples / total_samples))

        perfs = []
        logger.info('Evaluation after task {}:'.format(task_id))
        for i in range(task_id+1):
            _, val, test = tasks_data[i]
            val_perf = evaluate_model(model, x, y, val, task_id=i)
            test_perf = evaluate_model(model, x, y, test, task_id=i)
            perfs.append((val_perf, test_perf))
            logger.info('\tT n°{} - Val:{}%, test:{}%'.format(i, val_perf, test_perf))

if __name__ == '__main__':
    main(vars(get_args()))
