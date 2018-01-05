import argparse
import subprocess

import torch as t
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim import Adam

from dataloader import Dataloader
from model import Transormer
from nn.utils import ScheduledOptim

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='inf')
    parser.add_argument('--num-iterations', type=int, default=250_000, metavar='NI',
                        help='num iterations (default: 250_000)')
    parser.add_argument('--steps', type=int, default=5, metavar='S',
                        help='num steps before optimization step (default: 5)')
    parser.add_argument('--batch-size', type=int, default=20, metavar='BS',
                        help='batch size (default: 20)')
    parser.add_argument('--num-threads', type=int, default=4, metavar='BS',
                        help='num threads (default: 4)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='D',
                        help='dropout rate (default: 0.1)')
    parser.add_argument('--save', type=str, default='trained_model', metavar='TS',
                        help='path where save trained model to (default: "trained_model")')
    parser.add_argument('--tensorboard', type=str, default='default_tb', metavar='TB',
                        help='Name for tensorboard model')
    args = parser.parse_args()

    writer = SummaryWriter(args.tensorboard)

    t.set_num_threads(args.num_threads)
    loader = Dataloader('./dataloader/data/')

    model = Transormer(loader.vocab_size, loader.max_len, loader.pad_idx,
                       layers=6, heads=8, h_size=512, k_size=64, drop=args.dropout)
    if args.use_cuda:
        model = model.cuda()

    optimizer = ScheduledOptim(Adam(model.learnable_parameters(), betas=(0.9, 0.98), eps=1e-9), 512, 4000)

    crit = nn.CrossEntropyLoss(size_average=False, ignore_index=loader.pad_idx['ru'])

    print('Model have initialized')

    for i in range(args.num_iterations):

        optimizer.zero_grad()
        optimizer.update_learning_rate()

        out = 0
        for step in range(args.steps):
            condition, input, target = loader.torch(args.batch_size, 'train', args.use_cuda, volatile=False)

            nll = model.loss(condition, input, target, crit)
            nll /= args.steps
            out += nll.cpu().data

            nll.backward()

        optimizer.step()

        if i % 25 == 0:
            condition, input, target = loader.torch(args.batch_size * 4, 'valid', args.use_cuda, volatile=True)

            nll = model.loss(condition, input, target, crit)
            nll = nll.cpu().data

            writer.add_scalar('nll', nll, i)
            print('i {}, nll {}'.format(i, nll.numpy()))
            print('_________')

        if i % 500 == 0:
            condition, _, target = loader.torch(1, 'valid', args.use_cuda, volatile=True)
            indexes = ' '.join(map(str, condition[0].cpu().data.numpy()))
            subprocess.Popen(
                'echo "{}" | spm_decode --model=./dataloader/data/en.model --input_format=id'.format(indexes),
                shell=True
            )
            print('_________')
            indexes = ' '.join(map(str, target[0].cpu().data.numpy()[1:]))
            subprocess.Popen(
                'echo "{}" | spm_decode --model=./dataloader/data/ru.model --input_format=id'.format(indexes),
                shell=True
            )
            print('_________')
            indexes = model.translate(condition, loader, max_len=30, n_beams=10)
            subprocess.Popen(
                'echo "{}" | spm_decode --model=./dataloader/data/ru.model --input_format=id'.format(indexes),
                shell=True
            )
            print('_________')
