import gin
from torch import nn
import torch
import numpy as np


@gin.configurable
class Autoencoder(nn.Module):
    """Abstract autoencoder class."""
    def __init__(self, input_output_shape, **kwargs):
        super(Autoencoder, self).__init__()
        self.input_output_shape = input_output_shape
        if len(self.input_output_shape) == 0:
            self.input_output_shape = (1,)

    def encode(self, x):
        return NotImplementedError

    def decode(self, x):
        return NotImplementedError

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

@gin.configurable
class ModelAutoencoder(Autoencoder):
    """Torch model autoencoder."""
    def __init__(self, model_cls, flatten=True, **kwargs):
        super(ModelAutoencoder, self).__init__(**kwargs)
        self.flat_shape = (np.prod(self.input_output_shape),)
        self.flatten = flatten
        self.net_shape = self.flat_shape if flatten else self.input_output_shape
        self.model = model_cls(input_shape=self.net_shape,
                               output_shape=self.net_shape)


    def forward(self, x):
        if self.flatten:
            orig_shape = x.shape
            x = torch.flatten(x, start_dim=1)
        x = self.model(x)
        if self.flatten:
            x = x.reshape(orig_shape)
        return x