import torch
import gin
from torch import nn


@gin.configurable
class LagrangeMultipliers(nn.Module):
    """Learnable 1d tensor."""
    def __init__(self, n, param_min=-10, param_max=10, **kwargs):
        super(LagrangeMultipliers, self).__init__()

        self.n = n
        self.param_min = param_min
        self.param_max = param_max
        self.tensor = torch.nn.Parameter(torch.rand(n, dtype=torch.float32))

    def forward(self):
        self.project()
        return torch.exp(self.tensor)

    def project(self):
        self.tensor.data[:] = torch.clamp(self.tensor.data, min=self.param_min, max=self.param_max)
