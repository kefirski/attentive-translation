import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    def __init__(self, dim=1):
        super(GLU, self).__init__()

        self.dim = dim

    def forward(self, input):
        return F.glu(input, dim=self.dim)
