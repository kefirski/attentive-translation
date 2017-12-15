import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from .layer_norm import LayerNorm


class ContextWiseNN(nn.Module):
    def __init__(self, size, inner_size, autoregressive=False, dropout=0.1):
        super(ContextWiseNN, self).__init__()

        self.autoregressive = autoregressive
        self.padding_size = 1 if not autoregressive else 2

        self.conv = nn.ModuleList([
            weight_norm(nn.Conv1d(size, 2 * inner_size, 3, padding=self.padding_size, bias=False)),
            weight_norm(nn.Conv1d(inner_size, size, 3, padding=self.padding_size, bias=False))
        ])

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(size)

    def forward(self, input):
        residual = input

        input = input.transpose(1, 2)

        result = self.conv[0](input)
        if self.autoregressive:
            result = result[:, :, :-2]
        result = F.glu(result, dim=1)

        result = self.conv[1](result)
        if self.autoregressive:
            result = result[:, :, :-2]

        result = result.transpose(1, 2)

        result = self.dropout(result)
        return self.layer_norm(result + residual)
