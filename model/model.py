import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from dataloader import *
from nn.embedding import PositionalEmbeddings
from nn.transformer import Encoder, Decoder


class Transormer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, layers, heads, h_size, k_size, drop):
        """
        :param heads: Number of attention heads
        :param h_size: hidden size of input
        :param k_size: size of projected queries and keys
        :param drop: drop prob
        """
        super(Transormer, self).__init__()

        self.vocab_size = vocab_size

        self.embeddings = PositionalEmbeddings(vocab_size, max_seq_len, h_size)

        self.encoder = Encoder(layers, heads, h_size, k_size, k_size, drop)
        self.decoder = Decoder(layers, heads, h_size, k_size, k_size, drop)

        self.out_fc = nn.Sequential(
            weight_norm(nn.Linear(h_size, 4 * h_size)),
            nn.SELU(),
            weight_norm(nn.Linear(4 * h_size, vocab_size))
        )

    def forward(self, condition, input):
        """
        :param condition: An long tensor with shape of [batch_size, condition_len]
        :param input: An long tensor with shape of [batch_size, input_len]
        :return: An float tensor with shape of [batch_size, input_len, vocab_size]
        """

        batch_size, seq_len = input.size()

        condition_mask = t.eq(condition, 0).data
        input_mask = t.eq(input, 0).data

        condition = self.embeddings(condition)
        input = self.embeddings(input)

        condition = self.encoder(condition, condition_mask)
        out = self.decoder(input, condition, input_mask, condition_mask)

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
        out = out.view(-1, self.vocab_size)
        target = target.view(-1)

        nll = criterion(out, target) / batch_size

        return nll

    def generate(self, condition, loader: Dataloader, max_len=200, n_beams=35):

        self.eval()

        use_cuda = condition.is_cuda

        condition = self.embeddings(condition)
        condition, _ = self.encoder(condition)

        input = loader.go_input(1, use_cuda)
        input = self.embeddings(input)

        '''
        Starting point for beam search.
        Generate n_beams tokens
        '''
        out = self.decoder(input, condition)
        out = out.view(1, -1)
        out = F.softmax(self.out_fc(out).squeeze(0), dim=0).data.cpu().numpy()
        beams = loader.sample_char(out, n_beams)

        condition = condition.repeat(n_beams, 1, 1)

        for _ in range(max_len - 1):

            input = loader.to_tensor([beam.data for beam in beams], use_cuda)
            input = self.embeddings(input)

            out = self.decoder(input, condition)
            out = out[:, -1]
            out = F.softmax(self.out_fc(out), dim=1).data.cpu().numpy()

            beams = loader.beam_update(beams, out)

            if all([beam.data[-1] == loader.stop_token for beam in beams]):
                break

        return beams[-1].data

    def learnable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p
