import torch as t
import torch.nn as nn

from .position_wise_nn import PositionWiseNN
from ..attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, out, n_heads, h_size, k_size, v_size, dropout=0.1):
        """
        :param n_heads: Number of attention heads
        :param h_size: hidden size of input
        :param k_size: size of projected queries and keys
        :param v_size: size of projected values
        :param dropout: drop prob
        """
        super(DecoderLayer, self).__init__()

        self.out = out

        self.self_attention = MultiHeadAttention(n_heads, h_size, k_size, v_size, dropout)
        self.out_attention = MultiHeadAttention(n_heads, h_size, k_size, v_size, dropout) if out else None
        self.position_wise = PositionWiseNN(h_size, h_size * 4, dropout)

    def forward(self, decoder_input, condition, self_mask=None, out_mask=None):
        """
        :param decoder_input: An float tensor with shape of [batch_size, decoder_len, h_size]
        :param condition: An float tensor with shape of [batch_size, encoder_len, h_size]
        :param self_mask: An byte tensor with shape of [batch_size, decoder_len, decoder_len]
        :param out_mask: An byte tensor with shape of [batch_size, decoder_len, encoder_len]
        :return: An float tensor with shape of [batch_size, seq_len, h_size]
        """

        out, _ = self.self_attention(q=decoder_input, k=decoder_input, v=decoder_input, mask=self_mask)
        if self.out:
            out, _ = self.out_attention(q=out, k=condition, v=condition, mask=out_mask)
        out = self.position_wise(out)

        return out


class Decoder(nn.Module):
    def __init__(self, n_layers, n_heads, h_size, k_size, v_size, dropout=0.1):
        """
        :param n_heads: Number of attention heads
        :param h_size: hidden size of input
        :param k_size: size of projected queries and keys
        :param v_size: size of projected values
        :param dropout: drop prob
        """
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(i > 4, n_heads, h_size, k_size, v_size, dropout)
            for i in range(n_layers)
        ])

    def forward(self, decoder_input, condition, decoder_mask=None, condition_mask=None):
        """
        :param decoder_input: An float tensor with shape of [batch_size, decoder_len, h_size]
        :param condition: An float tensor with shape of [batch_size, encoder_len, h_size]
        :param decoder_mask: An byte tensor with shape of [batch_size, decoder_len, decoder_len]
        :param condition_mask: An byte tensor with shape of [batch_size, encoder_len]
        :return: An float tensor with shape of [batch_size, seq_len, h_size]
        """

        batch_size, decoder_len, _ = decoder_input.size()
        decoder_mask = t.eq(decoder_input.abs().sum(2), 0).repeat(1, decoder_len)\
            .view(batch_size, decoder_len, -1).data

        autogressive_mask = self.autogressive_mask(batch_size, decoder_len, decoder_input.is_cuda)
        decoder_mask += autogressive_mask
        decoder_mask = t.ge(decoder_mask, 1)

        condition_mask = t.eq(condition.abs().sum(2), 0).data
        condition_mask = condition_mask.repeat(1, decoder_len).view(batch_size, decoder_len, -1)

        out = decoder_input
        for layer in self.layers:
            out = layer(out, condition, decoder_mask, condition_mask)

        return out, []

    @staticmethod
    def autogressive_mask(batch_size, length, cuda):
        mask = t.ones(length, length).tril_(-1).byte()
        result = mask.transpose(0, 1).repeat(batch_size, 1).view(batch_size, length, length)
        if cuda:
            result = result.cuda()
        return result
