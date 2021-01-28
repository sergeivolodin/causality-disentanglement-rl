import gin
import numpy as np
from sparse_causal_model_learner_rl.metrics import find_value


# history of metrics
METRIC_CACHE = {}

@gin.configurable
def smooth(orig_key, now_epoch_info, smooth_steps=20, do_log=True,
           do_median=True,
           **kwargs):
    """Smooth a metric with windowed average."""
    if orig_key not in METRIC_CACHE:
        METRIC_CACHE[orig_key] = []
    else:
        METRIC_CACHE[orig_key] = METRIC_CACHE[orig_key][-smooth_steps:]

    metric = find_value(now_epoch_info, orig_key)
    if metric is not None:
        METRIC_CACHE[orig_key].append(metric)

    if METRIC_CACHE[orig_key]:
        data = METRIC_CACHE[orig_key]
        agg_fcn = np.median if do_median else np.mean
        if do_log:
            averaged = np.exp(agg_fcn(np.log(data)))
        else:
            averaged = agg_fcn(data)
        return averaged
    return None