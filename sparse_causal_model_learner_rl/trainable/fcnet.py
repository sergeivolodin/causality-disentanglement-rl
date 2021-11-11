from abc import ABC
import torch
from torch import nn
import gin

@gin.configurable
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

ReLU = gin.configurable(nn.ReLU)
LeakyReLU = gin.configurable(nn.LeakyReLU)
Tanh = gin.configurable(nn.Tanh)
Sigmoid = gin.external_configurable(nn.Sigmoid)
Linear = gin.external_configurable(nn.Linear)


def build_activation(cls, features=None):
    """Build an activation function with parameters."""
    if hasattr(cls, 'GIVE_N_FEATURES'):
        return cls(features=features)
    else:
        return cls()


@gin.configurable
class Scaler(nn.Module):
    """Scale output with learnable weights."""
    def __init__(self, shape):
        super().__init__()

        self.scale = nn.Parameter(torch.Tensor(*shape))
        self.loc = nn.Parameter(torch.Tensor(*shape))
        self.shape = shape

        torch.nn.init.ones_(self.scale)
        torch.nn.init.zeros_(self.loc)

    def forward(self, x):
        return x * self.scale + self.loc

    def __repr__(self):
        return f"Scaler({self.shape})"

@gin.configurable
class IdentityNet(nn.Module):
    """Return the input"""

    def __init__(self, input_shape, output_shape, **kwargs):
        super(IdentityNet, self).__init__()
        assert input_shape == output_shape, (input_shape, output_shape)

    def forward(self, x):
        return x

@gin.configurable
class FCNet(nn.Module):
    """Fully-connected neural network."""
    def __init__(self, input_shape, output_shape, hidden_sizes, activation_cls,
                 layers=nn.Linear,
                 add_scaler=False, add_input_batchnorm=False):
        super().__init__()

        assert len(input_shape) == 1
        assert len(output_shape) == 1

        self.input_dim = input_shape[0]
        self.output_dim = output_shape[0]
        self.hidden_sizes = hidden_sizes

        self.act_dims = self.hidden_sizes + [self.output_dim]
        if callable(activation_cls):
            self.activation = [build_activation(activation_cls, features=f) for f in self.hidden_sizes] + [None]
        elif isinstance(activation_cls, list):
            self.activation = [build_activation(act_cls, features=f)
                               if act_cls is not None else None
                               for f, act_cls in zip(self.act_dims, activation_cls)]
        elif activation_cls is None:
            self.activation = [None] * (len(self.hidden_sizes) + 1)
        else:
            raise NotImplementedError

        assert len(self.activation) == len(self.hidden_sizes) + 1, (self.activation,
                                                                    self.hidden_sizes)

        for i, act in enumerate(self.activation):
            if act is not None:
                setattr(self, 'act%02d' % (i + 1), act)

        self.dims = [self.input_dim] + self.hidden_sizes + [self.output_dim]
        if callable(layers):
            layers = [layers] * (len(self.dims) - 1)
        elif isinstance(layers, list):
            assert len(layers) == len(self.dims) - 1, (len(layers), self.dims - 1)
        else:
            raise NotImplementedError
        self.layers = layers

        self.fc = []
        for i in range(1, len(self.dims)):
            self.fc.append(self.layers[i - 1](in_features=self.dims[i - 1], out_features=self.dims[i]))

            # for torch to keep track of variables
            setattr(self, f'fc%02d' % i, self.fc[-1])

        if add_scaler:
            self.scaler = Scaler(shape=output_shape)
            
        if add_input_batchnorm:
            self.bn = torch.nn.BatchNorm1d(num_features=self.input_dim)

    def forward(self, x):
        if hasattr(self, 'bn'):
            x = self.bn(x)
        for i, fc in enumerate(self.fc):
            x = fc(x)
            if self.activation[i] is not None:
                x = self.activation[i](x)
        if hasattr(self, 'scaler'):
            x = self.scaler(x)
        return x
