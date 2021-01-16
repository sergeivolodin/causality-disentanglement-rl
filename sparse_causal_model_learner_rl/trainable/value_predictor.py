import gin
from torch import nn
from sparse_causal_model_learner_rl.trainable.fcnet import FCNet


@gin.configurable
class ValuePredictor(nn.Module):
    """Predicts value from features."""

    def __init__(self, feature_shape, **kwargs):
        self.feature_shape = feature_shape
        super(ValuePredictor, self).__init__()

    def forward(self, x):
        return NotImplementedError


@gin.configurable
class ModelValuePredictor(ValuePredictor):
    def __init__(self, model_cls=None, **kwargs):
        ValuePredictor.__init__(self, **kwargs)
        assert len(self.feature_shape) == 1
        self.model = model_cls(input_shape=self.feature_shape, output_shape=[1])

    def forward(self, x):
        return self.model(x)