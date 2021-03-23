import gin
import torch
import numpy as np
from torch import nn


@gin.configurable
class Reconstructor(nn.Module):
    """"""

    def __init__(self, observation_shape, feature_shape, **kwargs):
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

    def forward(self, x, **kwargs):
        return self.fc(x)

@gin.configurable
class IdentityReconstructor(Reconstructor):
    def __init__(self, **kwargs):
        super(IdentityReconstructor, self).__init__(**kwargs)
        assert self.observation_shape == self.feature_shape, (self.observation_shape, self.feature_shape)

    def forward(self, x, **kwargs):
        return x

@gin.configurable
class ModelReconstructor(Reconstructor):
    def __init__(self, model_cls=None, unflatten=False, give_source=False, **kwargs):
        super(ModelReconstructor, self).__init__(**kwargs)
        self.unflatten = unflatten
        self.give_source = give_source
        if self.unflatten:
            self.model_out_shape = (np.prod(self.observation_shape), )
        else:
            self.model_out_shape = self.observation_shape
        f_shape = self.feature_shape
        if self.give_source:
            f_shape = (f_shape[0] + 1,)
        self.model = model_cls(input_shape=f_shape, output_shape=self.model_out_shape)

    def forward(self, x, source_index=0):
        if self.give_source:
            x_source = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype, requires_grad=False) * source_index
            x = torch.cat([x, x_source], dim=1)
        x = self.model(x)
        if self.unflatten:
            x = x.view(x.shape[0], *self.observation_shape)
        return x

    def __repr__(self, *args, **kwargs):
        orig = super(ModelReconstructor, self).__repr__(*args, **kwargs)
        return f"{orig} unflatten={self.unflatten} give_source={self.give_source} model_out_shape={self.model_out_shape} obs_shape={self.observation_shape} feature_shape={self.feature_shape}"
