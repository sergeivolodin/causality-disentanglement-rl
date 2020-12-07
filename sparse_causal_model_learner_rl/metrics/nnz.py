import gin
import numpy as np
from sparse_causal_model_learner_rl.metrics.graph_threshold import threshold_features, threshold_action


@gin.configurable
def nnz(model, eps=1e-3, **kwargs):
    val = 0
    thresholds = {'fc_features.weight': threshold_features(model),
                  'fc_action.weight': threshold_action(model)}

    for n, p in model.named_parameters():
        val += np.sum(np.abs(p.detach().cpu().numpy()) > thresholds.get(n, eps))
    return val
