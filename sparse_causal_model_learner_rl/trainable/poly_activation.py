import torch
from torch import nn
import numpy as np
import gin


@gin.configurable
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
        if isinstance(features, list) or isinstance(features, tuple):
            features = np.prod(features)
        self.features = features
        self.orig_act = orig_act_cls()
        init = np.zeros((max_degree + 1, features), dtype=np.float32)
        init[1, :] = 1.0
        init[2, :] = 0.001
        self.a = nn.Parameter(torch.from_numpy(init), requires_grad=True)

    def forward(self, x):
        xshape = x.shape
        x = x.flatten(start_dim=1)
        assert x.shape[1] == self.features, (x.shape, self.features)

        x = self.orig_act(x)
        x = x.view(x.shape[0], x.shape[1], 1)
        x = x.tile((1, 1, self.max_degree + 1))
        powers = torch.arange(start=0, end=self.max_degree + 1, dtype=x.dtype, device=x.device)
        powers = powers.view(1, 1, self.max_degree + 1)
        powers = powers.tile((x.shape[0], x.shape[1], 1))

        x_powers = torch.pow(x.flatten(), powers.flatten()).view(*x.shape)

        p_with_coeff = torch.einsum('bfd,df->bfd', x_powers, self.a)
        out = p_with_coeff.sum(dim=2)
        out = out.view(*xshape)
        return out
