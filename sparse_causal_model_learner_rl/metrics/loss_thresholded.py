from causal_util import WeightRestorer
from sparse_causal_model_learner_rl.metrics.graph_threshold import threshold_action, threshold_features
from torch import nn
import torch
import gin

@gin.configurable
def loss_thresholded(loss, trainables, model: nn.Module, context, eps=1e-3,
                     eps_out=1e-15, delta=False, **kwargs):
    """Compute a loss when models weights are thresholded."""

    def threshold(param, val=eps):
        param[torch.abs(param) < val] = eps_out
        return param

    value_orig = loss(**context).item() if delta else 0

    with WeightRestorer(models=list(trainables.values())):
        thresholds = {'fc_features.weight': threshold_features(**context),
                      'fc_action.weight': threshold_action(**context)}

        dct = model.state_dict()
        dct = {x: threshold(y, thresholds.get(x, eps)) for x, y in dct.items()}
        model.load_state_dict(dct)

        value = loss(**context).item()

    result = value - value_orig

    return result