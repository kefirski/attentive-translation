import torch.nn as nn

from .position_wise_nn import PositionWiseNN
from ..attention import MultiHeadAttention


class Encoder(nn.Module):
    def __init__(self, n_layers, n_heads, h_size, k_size, v_size, m_size, dropout=0.1):
        """
        :param n_heads: Number of attention heads
        :param h_size: hidden size of input
        :param k_size: size of projected queries and keys
        :param v_size: size of projected values
        :param dropout: drop prob
        """
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(n_heads, h_size, k_size, v_size, m_size, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, input, mask=None):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, h_size]
        :param mask: An byte tensor with shape of [batch_size, seq_len, seq_len]
        :return: An float tensor with shape of [batch_size, seq_len, h_size]
        """

        out = input
        for layer in self.layers:
            out = layer(out, mask)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, h_size, k_size, v_size, m_size, dropout=0.1):
        """
        :param n_heads: Number of attention heads
        :param h_size: hidden size of input
        :param k_size: size of projected queries and keys
        :param v_size: size of projected values
        :param dropout: drop prob
        """
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(n_heads, h_size, k_size, v_size, m_size, dropout)
        self.position_wise = PositionWiseNN(h_size, h_size * 4, dropout)

    def forward(self, input, mask=None):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, h_size]
        :param mask: An byte tensor with shape of [batch_size, seq_len, seq_len]
        :return: An float tensor with shape of [batch_size, seq_len, h_size]
        """

        '''
        EncoderLayer network is defined over self-attention layer, 
        thus q, k and v are all obtained as encoder input.
        '''

        out, _ = self.attention(q=input, k=input, v=input, mask=mask)
        out = self.position_wise(out)

        return out
