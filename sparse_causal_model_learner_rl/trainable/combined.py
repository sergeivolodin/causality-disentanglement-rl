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

    def __repr__(self):
        return f"CombinedLinearLayer(inf={self.in_features}, outf={self.out_features}, n_models={self.n_models})"

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
    def __init__(self, hidden_sizes, activation_cls=nn.ReLU,
                 input_reshape=False,
                 skipconns=None,
                 add_input_batchnorm=False,
                 **kwargs):
        self.hidden_sizes = hidden_sizes
        self.input_reshape = input_reshape

        if self.input_reshape:
            assert len(kwargs['output_shape']) == 1
            kwargs['n_models'] = kwargs['output_shape'][0]
            kwargs['output_shape'] = (1,)

        super(FCCombinedModel, self).__init__(**kwargs)

        if callable(activation_cls):
            self.activation = [activation_cls()] * len(self.hidden_sizes) + [None]
        elif isinstance(activation_cls, list):
            self.activation = [act_cls() if act_cls is not None else None
                               for act_cls in activation_cls]
        elif activation_cls is None:
            self.activation = [None] * (len(self.hidden_sizes) + 1)
        else:
            raise NotImplementedError

        if skipconns is None:
            skipconns = [False] * len(self.activation)
        self.skipconns = skipconns
        print(self.skipconns)

        assert len(self.activation) == len(self.hidden_sizes) + 1, (self.activation,
                                                                    self.hidden_sizes)
        self.dims = [self.input_dim] + self.hidden_sizes + [self.output_dim]
        print(self.dims, self.n_models)
        self.fc = []
        for i in range(1, len(self.dims)):
            self.fc.append(
                CombinedLinearLayer(in_features=self.dims[i - 1], out_features=self.dims[i],
                                    n_models=self.n_models))

            # for torch to keep track of variables
            setattr(self, f'fc%02d' % i, self.fc[-1])
        if add_input_batchnorm:
            self.bn = nn.BatchNorm1d(self.input_dim)

    def __repr__(self, *args, **kwargs):
        orig = super(FCCombinedModel, self).__repr__(*args, **kwargs)
        return f"{orig} input_dim={self.input_dim} output_dim={self.output_dim} skips={self.skipconns} act={self.activation} hidden_sizes={self.hidden_sizes}"

    def forward(self, x):
        if self.input_reshape:
            if hasattr(self, 'bn'):
                x = self.bn(x)

            x = x.view(*x.shape, 1).expand(*[-1] * len(x.shape), self.n_models)

        for i, fc in enumerate(self.fc):
            x_inp = x
            x = fc(x)
            if self.activation[i] is not None:
                x = self.activation[i](x)
            if self.skipconns[i]:
                x = x + x_inp
        assert x.shape[1] == self.output_dim, (x.shape, self.output_dim, self.n_models)
        assert x.shape[2] == self.n_models
        if self.output_dim == 1:
            x = x.view(x.shape[0], x.shape[2])
        return x
