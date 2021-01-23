import gin
import numpy as np
from sparse_causal_model_learner_rl.metrics.graph_threshold import threshold_features, threshold_action


@gin.configurable
def nnz(model, eps=1e-3, **kwargs):
    val = 0
    thresholds = {'f': threshold_features(model),
                  'a': threshold_action(model)}

    val = 0
    val += np.sum(model.Mf > thresholds.get('f', eps))
    val += np.sum(model.Ma > thresholds.get('a', eps))

    return val
