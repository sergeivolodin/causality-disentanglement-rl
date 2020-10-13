import gin
from torch import nn

@gin.configurable
class Decoder(nn.Module):
    """"""
    def __init__(self, observation_shape, feature_shape):
        self.observation_shape = observation_shape
        self.feature_shape = feature_shape
        super(Decoder, self).__init__()

    def forward(self, x):
        return NotImplementedError

@gin.configurable
class LinearDecoder(Decoder):
    def __init__(self, use_bias=False):
        super(LinearDecoder, self).__init__()
        self.use_bias = use_bias
        assert len(self.observation_shape) == 1
        assert len(self.feature_shape) == 1
        self.fc = nn.Linear(self.observation_shape[0], self.feature_shape[0],
                            bias=self.use_bias)

    def forward(self, x):
        return self.fc(x)