import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, size, num_layers, transpose=False):
        super(ResNet, self).__init__()

        self.num_layers = num_layers
        self.size = size

        self.conv = nn.ModuleList([
            nn.Sequential(
                self.conv3x3(size, transpose),
                nn.SELU(),

                self.conv3x3(size, transpose),
            )

            for _ in range(num_layers)
        ])

    def forward(self, input):

        batch_size = input.size(0)

        should_view = False
        if len(input.size()) == 2:
            input = input.view(batch_size, self.size, -1)
            should_view = True

        for layer in self.conv:
            input = layer(input) + input
            input = F.selu(input)

        return input.view(batch_size, -1) if should_view else input

    @staticmethod
    def conv3x3(size, transpose):

        if transpose:
            return nn.utils.weight_norm(
                nn.ConvTranspose1d(size, size, kernel_size=3, stride=1, padding=1, bias=False)
            )

        return nn.utils.weight_norm(
            nn.Conv1d(size, size, kernel_size=3, stride=1, padding=1, bias=False)
        )
