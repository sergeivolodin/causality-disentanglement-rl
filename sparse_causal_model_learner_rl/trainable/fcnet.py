from abc import ABC
import torch
from torch import nn
import gin

ReLU = gin.external_configurable(nn.ReLU)
Tanh = gin.external_configurable(nn.Tanh)
Sigmoid = gin.external_configurable(nn.Sigmoid)


@gin.configurable
class Scaler(nn.Module):
    """Scale output with learnable weights."""
    def __init__(self, units):
        super().__init__()

        self.scale = nn.Parameter(torch.Tensor(units,))
        self.loc = nn.Parameter(torch.Tensor(units,))

        torch.nn.init.ones_(self.scale)
        torch.nn.init.zeros_(self.loc)

    def forward(self, x):
        return x * self.scale + self.loc


@gin.configurable
class FCNet(nn.Module):
    """Fully-connected neural network."""
    def __init__(self, input_shape, output_shape, hidden_sizes, activation_cls,
                 add_scaler=False):
        super().__init__()

        assert len(input_shape) == 1
        assert len(output_shape) == 1

        self.input_dim = input_shape[0]
        self.output_dim = output_shape[0]
        self.hidden_sizes = hidden_sizes
        if callable(activation_cls):
            self.activation = [activation_cls()] * len(self.hidden_sizes) + [None]
        elif isinstance(activation_cls, list):
            self.activation = [act_cls() if act_cls is not None else None
                               for act_cls in activation_cls]
        else:
            raise NotImplementedError

        assert len(self.activation) == len(self.hidden_sizes) + 1, (self.activation,
                                                                    self.hidden_sizes)
        self.dims = [self.input_dim] + self.hidden_sizes + [self.output_dim]
        self.fc = []
        for i in range(1, len(self.dims)):
            self.fc.append(nn.Linear(in_features=self.dims[i - 1], out_features=self.dims[i]))

            # for torch to keep track of variables
            setattr(self, f'fc%02d' % i, self.fc[-1])

        if add_scaler:
            self.scaler = Scaler(units=self.output_dim)

    def forward(self, x):
        for i, fc in enumerate(self.fc):
            x = fc(x)
            if self.activation[i] is not None:
                x = self.activation[i](x)
        if hasattr(self, 'scaler'):
            x = self.scaler(x)
        return x
