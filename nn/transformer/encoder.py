import torch as t
import torch.nn as nn
from torch.nn.utils import weight_norm

from .context_wise_nn import ContextWiseNN
from ..attention import MultiHeadAttention
from ..conv import GLUResNet, GLU


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

        self.conv = nn.Sequential(
            weight_norm(nn.Conv1d(h_size, 2 * h_size, 5, 1, padding=1, dilation=1, bias=False)),
            GLU(),

            GLUResNet(h_size, 2),

            weight_norm(nn.Conv1d(h_size, 2 * h_size, 5, stride=1, padding=2, bias=False)),
            GLU(),

            weight_norm(nn.Conv1d(h_size, 2 * h_size, 5, stride=2, padding=2, bias=False)),
            GLU(),

            weight_norm(nn.Conv1d(h_size, 2 * h_size, 5, stride=2, padding=2, bias=False)),
            GLU(),

            weight_norm(nn.Conv1d(h_size, 2 * h_size, 5, stride=2, padding=2, bias=False)),
            GLU(),

            GLUResNet(h_size, 2),

            weight_norm(nn.Conv1d(h_size, 2 * h_size, 5, stride=1, padding=2, bias=False)),
            GLU(),

            GLUResNet(h_size, 3)
        )

        self.layers = nn.ModuleList([
            EncoderLayer(n_heads, h_size, k_size, v_size, m_size, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, h_size]
        :return: An float tensor with shape of [batch_size, hidden_len, h_size]
        """

        input = input.transpose(1, 2)
        input = self.conv(input)
        input = input.transpose(1, 2)

        mask = t.eq(input.abs().sum(2), 0).data
        mask_app = mask.unsqueeze(1).repeat(1, mask.size(1), 1)

        out = input
        for layer in self.layers:
            out = layer(out, mask_app)

        return out, mask


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
        self.wise = ContextWiseNN(h_size, h_size * 2, dropout)

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
        out = self.wise(out)

        return out
