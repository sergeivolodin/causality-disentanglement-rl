import os
import numpy as np
import torch
from torch import nn
import gin


@gin.configurable
class AbstractCombinedModel(nn.Module):
    def __init__(self, n_models, input_shape, output_shape):
        super(AbstractCombinedModel, self).__init__()
        assert len(input_shape) == 1, input_shape
        assert len(output_shape) == 1, output_shape
        self.n_models = n_models
        self.input_dim = input_shape[0]
        self.output_dim = output_shape[0]


@gin.configurable
class CombinedLinearLayer(nn.Module):
    """Compute many linear layers of a single shape in a single pass.

    Input shape: [batch_dim, in_features, n_models]
    Output shape: [batch_dim, out_features, n_models]

    Equation (for one model): y = Wx+b
    """

    def __init__(self, in_features, out_features, n_models):
        super(CombinedLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_models = n_models
        self.weight = nn.Parameter(torch.zeros(self.out_features,
                                               self.in_features,
                                               self.n_models))
        self.bias = nn.Parameter(torch.zeros(self.out_features, self.n_models))
        self.reset_parameters()

    def weight_by_model(self, idx):
        return self.weight[:, :, idx]

    def bias_by_model(self, idx):
        return self.bias[:, idx]

    def reset_parameters(self, apply_fcn=nn.Linear.reset_parameters):
        class Resetter(nn.Module):
            def __init__(self, w, b):
                super(Resetter, self).__init__()
                self.weight = w
                self.bias = b

        for m in range(self.n_models):
            obj = Resetter(self.weight_by_model(m),
                           self.bias_by_model(m))
            apply_fcn(obj)

    def forward(self, x):
        w, b = self.weight, self.bias
        x = torch.einsum('bim,oim->bom', x, w) + b.view(1, *b.shape)
        return x


@gin.configurable
class FCCombinedModel(AbstractCombinedModel):
    def __init__(self, hidden_sizes, activation_cls=nn.ReLU, **kwargs):
        super(FCCombinedModel, self).__init__(**kwargs)
        self.hidden_sizes = hidden_sizes
        if callable(activation_cls):
            self.activation = [activation_cls()] * len(self.hidden_sizes) + [None]
        elif isinstance(activation_cls, list):
            self.activation = [act_cls() if act_cls is not None else None
                               for act_cls in activation_cls]
        elif activation_cls is None:
            self.activation = [None] * (len(self.hidden_sizes) + 1)
        else:
            raise NotImplementedError

        assert len(self.activation) == len(self.hidden_sizes) + 1, (self.activation,
                                                                    self.hidden_sizes)
        self.dims = [self.input_dim] + self.hidden_sizes + [self.output_dim]
        self.fc = []
        for i in range(1, len(self.dims)):
            self.fc.append(
                CombinedLinearLayer(in_features=self.dims[i - 1], out_features=self.dims[i],
                                    n_models=self.n_models))

            # for torch to keep track of variables
            setattr(self, f'fc%02d' % i, self.fc[-1])

    def forward(self, x):
        for i, fc in enumerate(self.fc):
            x = fc(x)
            if self.activation[i] is not None:
                x = self.activation[i](x)
        assert x.shape[1] == self.output_dim
        assert x.shape[2] == self.n_models
        if self.output_dim == 1:
            x = x.view(x.shape[0], x.shape[2])
        return x