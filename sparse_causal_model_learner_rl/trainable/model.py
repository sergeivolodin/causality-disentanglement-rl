import gin
from torch import nn
import numpy as np
import torch


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
        return list(self.fc_features.parameters())[0].detach().cpu().numpy()

    @property
    def Ma(self):
        """Return action model."""
        return list(self.fc_action.parameters())[0].detach().cpu().numpy()

@gin.configurable
class ModelModel(Model):
    """Model of the environment which uses a torch model to make predictions."""
    def __init__(self, model_cls, skip_connection=False, **kwargs):
        super(ModelModel, self).__init__(**kwargs)
        assert len(self.feature_shape) == 1, f"Features must be scalar: {self.feature_shape}"
        assert len(self.action_shape) == 1, f"Actions must be scalar: {self.action_shape}"
        self.model = model_cls(input_shape=(self.feature_shape[0] + self.action_shape[0], ),
                               output_shape=self.feature_shape)
        self.skip_connection = skip_connection

    def forward(self, f_t, a_t):
        fa_t = torch.cat((f_t, a_t), dim=1)
        f_t1 = self.model(fa_t)
        if self.skip_connection:
            f_t1 += f_t
        return f_t1
