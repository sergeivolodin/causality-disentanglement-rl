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
    def __init__(self, use_bias=False, use_batchnorm=False, **kwargs):
        super(LinearDecoder, self).__init__(**kwargs)
        self.use_bias = use_bias
        assert len(self.observation_shape) == 1
        assert len(self.feature_shape) == 1
        self.fc = nn.Linear(self.observation_shape[0], self.feature_shape[0],
                            bias=self.use_bias)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn = nn.BatchNorm1d(self.feature_shape[0], affine=False)

    def forward(self, x):
        x = self.fc(x)
        if self.use_batchnorm:
            x = self.bn(x)
        return x

@gin.configurable
class IdentityDecoder(Decoder):
    def __init__(self, **kwargs):
        super(IdentityDecoder, self).__init__(**kwargs)
        assert self.observation_shape == self.feature_shape

    def forward(self, x):
        return x

@gin.configurable
class ModelDecoder(Decoder):
    def __init__(self, model_cls, use_batchnorm=False, **kwargs):
        super(ModelDecoder, self).__init__(**kwargs)
        assert len(self.observation_shape) == 1
        assert len(self.feature_shape) == 1
        self.model = model_cls(input_shape=self.observation_shape, output_shape=self.feature_shape)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn = nn.BatchNorm1d(self.feature_shape[0], affine=False)

    def forward(self, x):
        x = self.model(x)
        if self.use_batchnorm:
            x = self.bn(x)
        return x