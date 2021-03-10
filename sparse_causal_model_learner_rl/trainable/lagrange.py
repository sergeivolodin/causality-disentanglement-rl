import torch
import gin
from torch import nn


@gin.configurable
class LagrangeMultipliers(nn.Module):
    """Learnable 1d tensor."""
    def __init__(self, n, **kwargs):
        super(LagrangeMultipliers, self).__init__()

        self.n = n
        self.tensor = torch.nn.Parameter(torch.rand(n, dtype=torch.float32))

    def forward(self):
        self.project()
        return self.tensor

    def project(self):
        self.tensor.data[:] = torch.clamp(self.tensor.data, min=0)