from abc import ABC

from torch import nn
import gin

ReLU = gin.external_configurable(nn.ReLU)
Sigmoid = gin.external_configurable(nn.Sigmoid)


@gin.configurable
class FCNet(nn.Module):
    """Fully-connected neural network."""
    def __init__(self, input_shape, output_shape, hidden_sizes, activation_cls):
        super().__init__()

        assert len(input_shape) == 1
        assert len(output_shape) == 1

        self.input_dim = input_shape[0]
        self.output_dim = output_shape[0]
        self.hidden_sizes = hidden_sizes
        self.activation = activation_cls()

        self.dims = [self.input_dim] + self.hidden_sizes + [self.output_dim]
        self.fc = []
        for i in range(1, len(self.dims)):
            self.fc.append(nn.Linear(in_features=self.dims[i - 1], out_features=self.dims[i]))

    def forward(self, x):
        for i, fc in enumerate(self.fc):
            x = fc(x)
            if i < len(self.fc) - 1:
                x = self.activation(x)
        return x
