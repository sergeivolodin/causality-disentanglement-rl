import gin
import numpy as np
from torch import nn
import torch


@gin.configurable
class Decoder(nn.Module):
    """"""

    def __init__(self, observation_shape, feature_shape, **kwargs):
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
        assert self.observation_shape == self.feature_shape, (self.observation_shape, self.feature_shape)

    def forward(self, x):
        return x

@gin.configurable
class ModelDecoder(Decoder):
    def __init__(self, model_cls, use_batchnorm=False, flatten=False, add_batch_number=False, **kwargs):
        super(ModelDecoder, self).__init__(**kwargs)
        self.flatten = flatten
        self.add_batch_number = add_batch_number
        assert len(self.feature_shape) == 1

        if self.flatten:
            self.model_obs_shape = (np.prod(self.observation_shape),)
        else:
            self.model_obs_shape = self.observation_shape

        self.model = model_cls(input_shape=self.model_obs_shape, output_shape=self.feature_shape)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn = nn.BatchNorm1d(self.feature_shape[0], affine=True)

    def forward(self, x):
        if self.flatten:
            x = x.flatten(start_dim=1)
        x = self.model(x)
        if self.use_batchnorm:
            x = self.bn(x)

        if self.add_batch_number:
            xconst = torch.linspace(start=-1, end=1, steps=x.shape[0], device=x.device, dtype=x.dtype, requires_grad=False).view(x.shape[0], 1)
            x = torch.cat([x, xconst], dim=1)

        return x

    def __repr__(self, *args, **kwargs):
        orig = super(ModelDecoder, self).__repr__(*args, **kwargs)
        return f"{orig} flatten={self.flatten} model_obs_shape={self.model_obs_shape} obs_shape={self.observation_shape} feature_shape={self.feature_shape}"
