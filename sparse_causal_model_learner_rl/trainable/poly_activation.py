import torch
from torch import nn
import numpy as np


class PolyAct(nn.Module):
    """Polynomial activation function. y = poly(x, w), w is learned."""

    GIVE_N_FEATURES = True

    def __repr__(self, *args, **kwargs):
        orig = super(PolyAct, self).__repr__(*args, **kwargs)
        return f"{orig} max_degree={self.max_degree} features={self.features}"

    def __init__(self, max_degree=3, orig_act_cls=nn.Tanh, features=None):
        super(PolyAct, self).__init__()
        # order: constant, x, x^2, ...
        self.max_degree = max_degree
        self.features = features
        self.orig_act = orig_act_cls()
        init = np.zeros((max_degree + 1, features), dtype=np.float32)
        init[1, :] = 1.0
        init[2, :] = 0.001
        self.a = nn.Parameter(torch.from_numpy(init), requires_grad=True)

    def forward(self, x):
        x = self.orig_act(x)
        powers = [torch.pow(x, i) for i in range(self.max_degree + 1)]
#         p_with_coeff = [torch.einsum('f,bf->bf', self.a[i, :], powers[i]) for i in range(self.max_degree + 1)]
        p_with_coeff = [powers[i] * self.a[i, :] for i in range(self.max_degree + 1)]
        return sum(p_with_coeff)
