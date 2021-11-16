import gin
from einops import rearrange
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
class SingleObjectPerChannelDecoder(nn.Module):
  """Get a model with x-axis, y-axis and sum filters."""
  def __init__(self, input_shape, output_shape):

    assert len(input_shape) == 3, input_shape
    assert len(output_shape) == 1, output_shape

    height, width, channels = input_shape
    num_out_features = channels * 3
    assert output_shape[0] == num_out_features

    super(SingleObjectPerChannelDecoder, self).__init__()

    self.ones = torch.ones(
        (height, width),
        dtype=torch.float32
    )

    def get_filter_y(height, width):
      """Get a matrix [[1, 2, ...], [1, 2, ...], ...]."""
      return torch.arange(
          1, 1 + width, dtype=torch.float32)\
      .unsqueeze(0).repeat(height, 1)

    self.filter_x = get_filter_y(width, height).T
    self.filter_y = get_filter_y(height, width)

    self.stacked = nn.Parameter(torch.stack([self.ones, self.filter_x, self.filter_y]))

  def forward(self, x):
    out = torch.einsum('bhwc, fhw -> bfc', x, self.stacked.detach())
    out = rearrange(out, 'b f c -> b ( f c )')
    return out


@gin.configurable
class SingleObjectWithLinear(nn.Module):
    def __init__(self, input_shape, intermediate_features, output_shape):
        self.const_dec = SingleObjectPerChannelDecoder(input_shape=input_shape, output_shape=(intermediate_features,))
        self.fc1 = nn.Linear(intermediate_features, output_shape[0])
    def forward(self, x):
        x = self.const_dec(x)
        x = self.fc1(x)
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
