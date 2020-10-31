import gin
from torch import nn
import numpy as np


@gin.configurable
class Model(nn.Module):
    def __init__(self, feature_shape, action_shape):
        super(Model, self).__init__()
        self.feature_shape = feature_shape
        if len(self.feature_shape) == 0:
            self.feature_shape = (1,)
        self.action_shape = action_shape
        if len(self.action_shape) == 0:
            self.action_shape = (1,)

    def forward(self, f_t, a_t):
        return NotImplementedError


@gin.configurable
class LinearModel(Model):
    def __init__(self, use_bias=True, **kwargs):
        super(LinearModel, self).__init__(**kwargs)
        self.use_bias = use_bias
        assert len(self.feature_shape) <= 1, f"Features must be scalar: {self.feature_shape}"
        assert len(self.action_shape) <= 1, f"Actions must be scalar: {self.action_shape}"
        self.fc_features = nn.Linear(self.feature_shape[0], self.feature_shape[0], bias=self.use_bias)
        self.fc_action = nn.Linear(self.action_shape[0], self.feature_shape[0], bias=False)

    def forward(self, f_t, a_t):
        f_next_f = self.fc_features(f_t)
        f_next_a = self.fc_action(a_t)
        return f_next_f + f_next_a

    @property
    def Mf(self):
        """Return features model."""
        return list(self.fc_features.parameters())[0].detach().numpy()

    @property
    def Ma(self):
        """Return action model."""
        return list(self.fc_action.parameters())[0].detach().numpy()