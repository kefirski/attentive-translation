import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from dataloader import *
from nn.embedding import PositionalEmbeddings
from nn.transformer import Encoder, Decoder


class Transormer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, n_layers, n_heads, h_size, k_size, v_size, m_size, dropout):
        """
        :param n_heads: Number of attention heads
        :param h_size: hidden size of input
        :param k_size: size of projected queries and keys
        :param v_size: size of projected values
        :param dropout: drop prob
        """
        super(Transormer, self).__init__()

        self.vocab_size = vocab_size

        self.embeddings = PositionalEmbeddings(vocab_size, max_seq_len, h_size)

        self.encoder = Encoder(n_layers, n_heads, h_size, k_size, v_size, m_size, dropout)
        self.decoder = Decoder(n_layers, n_heads, h_size, k_size, v_size, m_size, dropout)

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

        condition_mask = self.mask(condition)
        decoder_in_mask = self.mask(input)
        decoder_out_mask = self.mask(condition, repeat_size=seq_len)

        condition = self.embeddings(condition)
        input = self.embeddings(input)

        condition = self.encoder(condition, mask=condition_mask)
        out = self.decoder(input, condition, decoder_in_mask, decoder_out_mask)

        out = out.view(batch_size * seq_len, -1)
        out = self.out_fc(out).view(batch_size, seq_len, -1)

        return out

    @staticmethod
    def mask(tensor, repeat_size=None):

        _, seq_len = tensor.size()
        if repeat_size is None:
            repeat_size = seq_len

        mask = t.eq(tensor, 0).data
        mask = mask.repeat(1, repeat_size).view(-1, repeat_size, seq_len)

        return mask

    def loss(self, condition, input, target, criterion, eval=False):

        if eval:
            self.eval()
        else:
            self.train()

        out = self(condition, input)
        out = out.view(-1, self.vocab_size)
        target = target.view(-1)

        nll = criterion(out, target) / condition.size(0)

        return nll

    def generate(self, condition, loader: Dataloader, max_len=100, n_beams=4):

        self.eval()

        use_cuda = condition.is_cuda

        condition = self.embeddings(condition)
        condition = self.encoder(condition)

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

        return '\n'.join([beam.data for beam in beams])

    def learnable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p
