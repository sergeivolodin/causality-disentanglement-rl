import torch
import numpy as np
import gin
from torch import nn

@gin.configurable
class LagrangeMultipliers(nn.Module):
    """Learnable 1d tensor."""
    def __init__(self, n, param_min=-10, param_init=0.0, param_max=10, vectorized=False,
            max_second_dim=1000,
            fcn=None, **kwargs):
        super(LagrangeMultipliers, self).__init__()

        self.n = n
        self.fcn = fcn
        self.param_min = param_min
        self.param_max = param_max
        self.vectorized = vectorized
        if vectorized:
            self.initialized = np.full(shape=(n, max_second_dim), fill_value=False)
            self.tensor = torch.nn.Parameter(torch.ones(n, max_second_dim, dtype=torch.float32) * param_init)
        else:
            self.initialized = np.full(shape=(n,), fill_value=False)
            self.tensor = torch.nn.Parameter(torch.ones(n, dtype=torch.float32) * param_init)

    def state_dict(self, *args, **kwargs):
        orig = super(LagrangeMultipliers, self).state_dict(*args, **kwargs)
        orig['initialized'] = self.initialized
        return orig

    def load_state_dict(self, *args, **kwargs):
        orig = super(LagrangeMultipliers, self).load_state_dict(*args, **kwargs)
        if len(args) == 1 and 'initialized' in args[0]:
            self.initialized = args[0]['initialized']
        return orig

    def __str__(self):
        return f"LagrangeMultipliers(n={self.n} initialized={self.initialized} fcn={self.fcn} param_min={self.param_min} param_max={self.param_max} vectorized={self.vectorized})"

    def set_value(self, idx, val, component=0):
        if self.fcn == 'exp':
            val = np.log(np.abs(val))
        elif self.fcn == 'square':
            val = np.sqrt(np.abs(val))
        elif self.fcn == 'square_root':
            val = np.power(val, 2)
        elif self.fcn == 'identity':
            pass
        else:
            raise NotImplementedError(f"{self.fcn} {val}")

        if self.vectorized:
            self.tensor.data[idx, component] = float(val)
            self.initialized[idx, component] = True
            assert not self.initialized[idx, component]
        else:
            self.tensor.data[idx] = float(val)
            self.initialized[idx] = True
            assert not self.initialized[idx]

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
        elif self.fcn == 'identity':
            pass
        else:
            raise NotImplementedError
        return data

    def project(self):
        self.tensor.data[:] = torch.clamp(self.tensor.data, min=self.param_min, max=self.param_max)
