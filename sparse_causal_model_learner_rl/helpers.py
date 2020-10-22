import torch
import gin
import numpy as np


def flatten_dict_keys(dct, prefix='', separator='/'):
    """Nested dictionary to a flat dictionary."""
    result = {}
    for key, value in dct.items():
        if isinstance(value, dict):
            subresult = flatten_dict_keys(value, prefix=prefix + key + '/',
                                          separator=separator)
            result.update(subresult)
        else:
            result[prefix + key] = value
    return result


def torch_to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    if isinstance(x, np.ndarray):
        if len(x.shape) == 1 and x.shape[0] == 1:
            x = np.mean(x)
        elif len(x.shape) == 0:
            x = np.mean(x)
    return x

def postprocess_info(d):
    """Prepare the info before exporting to Sacred."""
    d = flatten_dict_keys(d)
    d = {x: torch_to_numpy(y) for x, y in d.items()}
    return d