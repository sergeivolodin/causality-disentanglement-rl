from causal_util import WeightRestorer
from torch import nn
import torch
import gin

@gin.configurable
def loss_thresholded(loss, trainables, model: nn.Module, context, eps=1e-3,
                     eps_out=1e-15, delta=False, **kwargs):
    """Compute a loss when models weights are thresholded."""

    def threshold(param):
        param[torch.abs(param) < eps] = eps_out
        return param

    if delta:
        value_orig = loss(**context).item()

    with WeightRestorer(models=list(trainables.values())):
        dct = model.state_dict()
        dct = {x: threshold(y) for x, y in dct.items()}
        model.load_state_dict(dct)

        value = loss(**context).item()

    if delta:
        result = value - value_orig
    else:
        result = value

    return result