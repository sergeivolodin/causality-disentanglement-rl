import os
import numpy as np
import torch
from torch import nn
import gin
from sparse_causal_model_learner_rl.trainable.fcnet import build_activation


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
class CombinedQuadraticLayer(CombinedLinearLayer):
    """Compute many quadratic layers of a single shape in a single pass.

    Input shape: [batch_dim, in_features, n_models]
    Output shape: [batch_dim, out_features, n_models]

    Equation (for one model): y = x^TAx+Wx+b
    """

    def __init__(self, **kwargs):
        super(CombinedQuadraticLayer, self).__init__(**kwargs)
        self.qweight = nn.Parameter(torch.zeros(self.out_features,
                                                self.in_features,
                                                self.in_features,
                                                self.n_models))
        self.reset_parameters()

    def __repr__(self):
        return f"CombinedQuadraticLayer(inf={self.in_features}, outf={self.out_features}, n_models={self.n_models})"

    def weight_by_model(self, idx):
        return self.weight[:, :, idx]

    def bias_by_model(self, idx):
        return self.bias[:, idx]

    def qweight_by_model(self, idx):
        return self.qweight[:, :, :, idx]

    def reset_parameters(self, apply_fcn=nn.Linear.reset_parameters, qscaler=0.01):
        super(CombinedQuadraticLayer, self).reset_parameters(apply_fcn=apply_fcn)
        if hasattr(self, 'qweight'):
            self.qweight.data = torch.randn(self.out_features, self.in_features, self.in_features, self.n_models) * qscaler

    def forward(self, x):
        out = super(CombinedQuadraticLayer, self).forward(x)
        out += torch.einsum('bim,bjm,oijm->bom', x, x, self.qweight)
        return out


@gin.configurable
class FCCombinedModel(AbstractCombinedModel):
    def __init__(self, hidden_sizes, activation_cls=nn.ReLU,
                 input_reshape=False,
                 layers=CombinedLinearLayer,
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

        self.act_dims = self.hidden_sizes + [self.output_dim]
        if callable(activation_cls):
            self.activation = [build_activation(activation_cls, features=f * self.n_models)
                              for f in self.act_dims[:-1]] + [None]
        elif isinstance(activation_cls, list):
            self.activation = [build_activation(act_cls, features=f * self.n_models)
                               if act_cls is not None else None
                               for f, act_cls in zip(self.act_dims, activation_cls)]
        elif activation_cls is None:
            self.activation = [None] * (len(self.hidden_sizes) + 1)
        else:
            raise NotImplementedError

        for i, act in enumerate(self.activation):
            if act is not None:
                setattr(self, 'act%d' % (i + 1), act)

        if skipconns is None:
            skipconns = [False] * len(self.activation)
        self.skipconns = skipconns
        print(self.skipconns)

        assert len(self.activation) == len(self.hidden_sizes) + 1, (self.activation,
                                                                    self.hidden_sizes)
        self.dims = [self.input_dim] + self.hidden_sizes + [self.output_dim]
        print(self.dims, self.n_models)
        self.fc = []

        if callable(layers):
            layers = [layers] * (len(self.dims) - 1)

        if isinstance(layers, list):
            assert len(layers) == len(self.dims) - 1, (len(layers), len(self.dims))
        else:
            raise NotImplementedError

        self.layers = layers

        for i in range(1, len(self.dims)):
            self.fc.append(self.layers[i - 1](
                in_features=self.dims[i - 1],
                out_features=self.dims[i],
                n_models=self.n_models))

            # for torch to keep track of variables
            setattr(self, f'fc%02d' % i, self.fc[-1])
        if add_input_batchnorm:
            self.bn = nn.BatchNorm1d(self.input_dim)

    def __repr__(self, *args, **kwargs):
        orig = super(FCCombinedModel, self).__repr__(*args, **kwargs)
        return f"{orig} input_dim={self.input_dim} output_dim={self.output_dim} skips={self.skipconns} act={self.activation} hidden_sizes={self.hidden_sizes} layers={self.layers}"

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
