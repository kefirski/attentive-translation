import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from dataloader import *
from nn.transformer import Encoder, Decoder


class Transormer(nn.Module):
    def __init__(self, vocab_size, max_len, pad_idx, layers, heads, h_size, k_size, drop):
        """
        :param heads: Number of attention heads
        :param h_size: hidden size of input
        :param k_size: size of projected queries and keys
        :param drop: drop prob
        """
        super(Transormer, self).__init__()

        self.vocab_size = vocab_size

        self.encoder = Encoder(vocab_size['en'], max_len['en'], pad_idx['en'],
                               layers, heads, h_size, k_size, k_size, drop)
        self.decoder = Decoder(vocab_size['ru'], max_len['ru'], pad_idx['ru'],
                               layers, heads, h_size, k_size, k_size, drop)

        self.out_fc = nn.Sequential(
            weight_norm(nn.Linear(h_size, 4 * h_size)),
            nn.SELU(),
            weight_norm(nn.Linear(4 * h_size, vocab_size['ru']))
        )

    def forward(self, condition, input):
        """
        :param condition: An long tensor with shape of [batch_size, condition_len]
        :param input: An long tensor with shape of [batch_size, input_len]
        :return: An float tensor with shape of [batch_size, input_len, vocab_size]
        """

        batch_size, seq_len = input.size()

        condition, mask = self.encoder(condition)
        out = self.decoder(input, condition, mask)

        out = out.view(batch_size * seq_len, -1)
        out = self.out_fc(out).view(batch_size, seq_len, -1)

        return out

    def loss(self, condition, input, target, criterion, eval=False):

        batch_size, *_ = condition.size()

        if eval:
            self.eval()
        else:
            self.train()

        out = self(condition, input)
        out = out.view(-1, self.vocab_size['ru'])
        target = target.view(-1)

        nll = criterion(out, target) / batch_size

        return nll

    def generate(self, condition, loader: Dataloader, max_len=80, n_beams=35):

        self.eval()

        use_cuda = condition.is_cuda

        condition, _ = self.encoder(condition)

        input = loader.go_input(1, use_cuda, lang='ru', volatile=True)

        '''
        Starting point for beam search.
        Generate n_beams tokens
        '''
        out = self.decoder(input, condition)
        out = out.view(1, -1)
        out = F.softmax(self.out_fc(out).squeeze(0), dim=0).data.cpu().numpy()
        beams = Beam.start_search(out, n_beams)

        condition = condition.repeat(n_beams, 1, 1)

        for _ in range(max_len):

            input = loader.to_tensor([beam.data for beam in beams], use_cuda, lang='ru', volatile=True)

            out = self.decoder(input, condition)
            out = out[:, -1]
            out = F.softmax(self.out_fc(out), dim=1).data.cpu().numpy()

            beams = Beam.update(beams, out)

            '''
            There is no reason to continiue beam search
            if all sequences had already emited stop symbol
            '''
            if all([any([idx == loader.stop_idx['ru'] for idx in beam.data]) for beam in beams]):
                break

        return ' '.join(beams[-1].data)

    def learnable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p
