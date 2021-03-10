import torch
import gin
from torch import nn


@gin.configurable
class LagrangeMultipliers(nn.Module):
    """Learnable 1d tensor."""
    def __init__(self, n):
        super(LagrangeMultipliers, self).__init__()

        self.tensor = torch.autograd.Variable(torch.rand(n, dtype=torch.float32),
                                              requires_grad=True)

    def forward(self):
        self.project()
        return self.tensor

    def project(self):
        self.tensor.data[:] = torch.clamp(self.tensor.data, min=0)