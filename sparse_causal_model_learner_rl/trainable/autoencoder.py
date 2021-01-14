import gin
from torch import nn


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
    def __init__(self, model_cls, **kwargs):
        super(ModelAutoencoder, self).__init__(**kwargs)
        self.model = model_cls(input_shape=self.input_output_shape,
                               output_shape=self.input_output_shape)


    def forward(self, x):
        return self.model(x)