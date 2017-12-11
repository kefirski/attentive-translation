import argparse

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
    parser.add_argument('--steps', type=int, default=1, metavar='S',
                        help='num steps before optimization step (default: 15)')
    parser.add_argument('--batch-size', type=int, default=2, metavar='BS',
                        help='batch size (default: 80)')
    parser.add_argument('--num-threads', type=int, default=4, metavar='BS',
                        help='num threads (default: 4)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--dropout', type=float, default=0.15, metavar='D',
                        help='dropout rate (default: 0.15)')
    parser.add_argument('--save', type=str, default='trained_model', metavar='TS',
                        help='path where save trained model to (default: "trained_model")')
    parser.add_argument('--tensorboard', type=str, default='default_tb', metavar='TB',
                        help='Name for tensorboard model')
    args = parser.parse_args()

    writer = SummaryWriter(args.tensorboard)

    t.set_num_threads(args.num_threads)
    loader = Dataloader('/Users/daniil/projects/atran/dataloader/data/')

    model = Transormer(loader.vocab_size, loader.max_len, 6, 8, 120, 25, 25, dropout=args.dropout)
    if args.use_cuda:
        model = model.cuda()

    optimizer = ScheduledOptim(Adam(model.learnable_parameters(), betas=(0.9, 0.98), eps=1e-9), 120, 5000)

    crit = nn.CrossEntropyLoss(size_average=False, ignore_index=0)

    print('Model have initialized')

    for i in range(args.num_iterations):

        out = 0
        for step in range(args.steps):
            condition, condition, target = loader.torch(args.batch_size, args.use_cuda, volatile=False)

            nll = model.loss(condition, condition, target, crit)
            nll /= args.steps
            out += nll.cpu().data

            nll.backward()

        optimizer.step()
        optimizer.zero_grad()

        if i % 25 == 0:
            print('i {}, nll {}'.format(i, out.numpy()))
            print('_________')

        if i % 50 == 0:
            condition, _, target = loader.torch(1, args.use_cuda, volatile=True)
            print(''.join([loader.idx_to_token[idx] for idx in condition[0].cpu().data.numpy()]))
            print('_________')
            print(''.join([loader.idx_to_token[idx] for idx in target[0].cpu().data.numpy()]))
            print('_________')
            print(model.generate(condition, loader, max_len=140, n_beams=5))
            print('_________')
