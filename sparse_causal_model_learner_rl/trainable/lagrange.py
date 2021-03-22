import torch
import gin
from torch import nn

@gin.configurable
class LagrangeMultipliers(nn.Module):
    """Learnable 1d tensor."""
    def __init__(self, n, param_min=-10, param_max=10, fcn=None, **kwargs):
        super(LagrangeMultipliers, self).__init__()

        self.n = n
        self.fcn = fcn
        self.param_min = param_min
        self.param_max = param_max
        self.tensor = torch.nn.Parameter(torch.ones(n, dtype=torch.float32) * param_max)

    def forward(self):
        self.project()

        data = self.tensor
        if callable(self.fcn):
            data = self.fcn(data)
        elif self.fcn == 'exp':
            data = torch.exp(data)
        elif self.fcn == 'square':
            data = torch.pow(data, 2.0)
        elif self.fcn == 'square_root':
            data = torch.sqrt(data)
        return data

    def project(self):
        self.tensor.data[:] = torch.clamp(self.tensor.data, min=self.param_min, max=self.param_max)
