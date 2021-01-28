import gin
import numpy as np

# history of metrics
METRIC_CACHE = {}

def metric_by_flattened_key(dct, key, separator='/'):
    """Get a metric by key with slashes from a nested dict."""
    if key in dct:
        return dct[key]
    elif separator in key:
        splitted = key.split(separator)
        return metric_by_flattened_key(dct[splitted[0]], separator.join(splitted[1:]))
    else:
        return None

@gin.configurable
def smooth(orig_key, now_epoch_info, smooth_steps=20, do_log=True, **kwargs):
    """Smooth a metric with windowed average."""
    if orig_key not in METRIC_CACHE:
        METRIC_CACHE[orig_key] = []
    else:
        METRIC_CACHE[orig_key] = METRIC_CACHE[orig_key][-smooth_steps:]

    metric = metric_by_flattened_key(now_epoch_info, orig_key)
    if metric is not None:
        METRIC_CACHE[orig_key].append(metric)

    if METRIC_CACHE[orig_key]:
        data = METRIC_CACHE[orig_key]
        if do_log:
            averaged = np.exp(np.mean(np.log(data)))
        else:
            averaged = np.mean(data)
        return averaged
    return None