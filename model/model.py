import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from nn.transformer import Encoder, Decoder
from nn.embedding import PositionalEmbeddings
from dataloader.beam import Beam


class Transormer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, n_layers=6, n_heads=8, h_size=120, k_size=25, v_size=25, dropout=0.1):
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

        self.encoder = Encoder(n_layers, n_heads, h_size, k_size, v_size, dropout)
        self.decoder = Decoder(n_layers, n_heads, h_size, k_size, v_size, dropout)

        self.out_fc = nn.Sequential(
            weight_norm(nn.Linear(h_size, 4 * h_size)),
            nn.SELU(),
            weight_norm(nn.Linear(4 * h_size, vocab_size))
        )

    def forward(self, input, decoder_input):
        """
        :param input: An long tensor with shape of [batch_size, encoder_len]
        :param decoder_input: An long tensor with shape of [batch_size, decoder_len]
        :return: An float tensor with shape of [batch_size, decoder_len, vocab_size]
        """

        input_mask = self.mask(input)
        decoder_input_mask = self.mask(decoder_input)
        condition_mask = self.mask(input, repeat_size=decoder_input.size(1))

        input = self.embeddings(input)
        decoder_input = self.embeddings(decoder_input)

        condition = self.encoder(input, mask=input_mask)
        out = self.decoder(decoder_input, condition, decoder_input_mask, condition_mask)

        batch_size, seq_len, _ = out.size()
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

    def loss(self, forward, input, decoder_input, target, criterion, eval=False):

        if eval:
            self.eval()
        else:
            self.train()

        out = forward(input, decoder_input)
        out = out.view(-1, self.vocab_size)
        target = target.view(-1)

        nll = criterion(out, target) / input.size(0)

        return nll

    def generate(self, input, loader, embeddings, cuda, max_len=30, n_beams=10):

        self.eval()

        input = embeddings(input)
        if cuda:
            input = input.cuda()

        condition = self.encoder(input)

        decoder_input = loader.go_input(1, False)
        decoder_embed = embeddings(decoder_input)
        if cuda:
            decoder_embed = decoder_embed.cuda()

        beams = [Beam() for _ in range(n_beams)]

        '''
        Starting point for beam search.
        Generate n_beams characters
        '''
        decoder_out, _ = self.decoder(decoder_embed, condition)
        decoder_out = decoder_out.view(1, -1)
        decoder_out = F.softmax(self.out_fc(decoder_out).squeeze(0), dim=0).data.cpu().numpy()
        samplings = loader.sample_char(decoder_out, n_beams)

        for i, (_, word, prob) in enumerate(samplings):
            beams[i].update(beams[i].prob * prob, word)

        condition = condition.repeat(n_beams, 1, 1)

        decoder_input = loader.to_tensor([beam.data for beam in beams], False)
        decoder_embed = embeddings(decoder_input)
        if cuda:
            decoder_embed = decoder_embed.cuda()

        for _ in range(max_len - 1):

            decoder_out, _ = self.decoder(decoder_embed, condition)
            decoder_out = decoder_out[:, -1]
            decoder_out = F.softmax(self.out_fc(decoder_out), dim=1).data.cpu().numpy()

            beams = loader.beam_update(beams, decoder_out, n_beams)

            decoder_input = loader.to_tensor([beam.data for beam in beams], False)
            decoder_embed = embeddings(decoder_input)
            if cuda:
                decoder_embed = decoder_embed.cuda()

        return '\n'.join([beam.data for beam in beams])

    def learnable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p