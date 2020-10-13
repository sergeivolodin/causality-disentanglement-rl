import gin
from torch import nn

@gin.configurable
class Reconstructor(nn.Module):
    """"""
    def __init__(self, observation_shape, feature_shape):
        self.observation_shape = observation_shape
        self.feature_shape = feature_shape
        super(Reconstructor, self).__init__()

    def forward(self, x):
        return NotImplementedError

@gin.configurable
class LinearReconstructor(Reconstructor):
    def __init__(self, use_bias=False, **kwargs):
        super(LinearReconstructor, self).__init__(**kwargs)
        self.use_bias = use_bias
        assert len(self.observation_shape) == 1
        assert len(self.feature_shape) == 1
        self.fc = nn.Linear(self.feature_shape[0], self.observation_shape[0],
                            bias=self.use_bias)

    def forward(self, x):
        return self.fc(x)