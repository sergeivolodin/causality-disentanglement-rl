import gin
from torch import nn


@gin.configurable
class Model(nn.Module):
    def __init__(self, feature_shape, action_shape):
        self.feature_shape = feature_shape
        self.action_shape = action_shape

    def forward(self, x):
        return NotImplementedError


@gin.configurable
class LinearModel(Model):
    def __init__(self, use_bias=True):
        super(LinearModel, self).__init__()
        self.use_bias = use_bias
        assert len(self.feature_shape) == 1
        assert len(self.action_shape) == 1
        self.fc_features = nn.Linear(self.feature_shape, self.feature_shape, bias=self.use_bias)
        self.fc_action = nn.Linear(self.action_shape, self.feature_shape, bias=False)

    def forward(self, x):
        f_t = x.get('features')
        a_t = x.get('action')
        f_next_f = self.fc_features(f_t)
        f_next_a = self.fc_action(a_t)
        return f_next_f + f_next_a