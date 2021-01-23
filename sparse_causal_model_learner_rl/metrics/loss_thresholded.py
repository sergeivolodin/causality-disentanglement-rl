from causal_util import WeightRestorer
from sparse_causal_model_learner_rl.metrics.graph_threshold import threshold_action, threshold_features
from torch import nn
import torch
import gin

@gin.configurable
def loss_thresholded(loss, trainables, model: nn.Module, context, eps=1e-3,
                     eps_out=1e-15, delta=False, use_gnn=False, **kwargs):
    """Compute a loss when models weights are thresholded."""

    def threshold(param, thresholds, param_name):

        if use_gnn:
            n_features = model.n_features
            param[:, :n_features][
                torch.abs(param[:, :n_features]) < thresholds.get('fc_features.weight', eps)] = eps_out
            param[:, n_features:][
                torch.abs(param[:, n_features:]) < thresholds.get('fc_action.weight', eps)] = eps_out
        else:
            param[torch.abs(param) < thresholds.get(param_name, eps)] = eps_out

        return param

    def maybe_unpack_dict(l):
        if isinstance(l, dict):
            return l['loss']
        return l

    value_orig = maybe_unpack_dict(loss(**context)).item() if delta else 0

    with WeightRestorer(models=list(trainables.values())):
        thresholds = {'fc_features.weight': threshold_features(**context),
                      'fc_action.weight': threshold_action(**context)}

        if use_gnn:
            dct = dict(model.sparsify_tensors())
            dct = {x: y.detach().cpu() for x, y in dct.items()}
        else:
            dct = model.state_dict()
        dct = {x: threshold(y, thresholds, x) for x, y in dct.items()}
        model.load_state_dict(dct, strict=not use_gnn)

        value = maybe_unpack_dict(loss(**context)).item()

    result = value - value_orig

    return result