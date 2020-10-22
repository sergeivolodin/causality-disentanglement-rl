from time import time

import gin
import gym
import numpy as np
import torch

def one_hot_encode(n, value):
    """Get a one-hot encoding of length n with a given value."""
    result = np.zeros(n, dtype=np.float32)
    assert isinstance(n, int) and isinstance(value, int)
    assert n >= 0 and 0 <= value < n
    result[value] = 1
    return result

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
    if isinstance(x, np.float32) or isinstance(x, np.float64):
        x = float(x)
    if isinstance(x, np.int32) or isinstance(x, np.int64):
        x = int(x)
    return x

def postprocess_info(d):
    """Prepare the info before exporting to Sacred."""
    d = flatten_dict_keys(d)
    d = {x: torch_to_numpy(y) for x, y in d.items()}
    return d

def rescale(x, min_value, max_value):
    """Rescale x to be in [-1, 1] assuming values in [min_value, max_value]."""
    x = np.copy(x)
    x[x < min_value] = min_value
    x[x > max_value] = max_value
    x = (x - min_value) / (max_value - min_value)
    x = 2 * (x - 0.5)
    return x


def vec_heatmap(vec, min_value=-1, max_value=1, sq_side=10):
    """Show heatmap for the vector."""
    vec = vec.flatten()
    n = vec.shape[0]
    vec_rescaled = rescale(vec, min_value, max_value)
    obs = np.zeros((sq_side, sq_side * n, 3))
    for i in range(n):
        square = obs[:, i * sq_side:(i + 1) * sq_side]
        val = vec_rescaled[i]
        if val > 0:
            square[:, :, 1] = vec_rescaled[i]
        elif val < 0:
            square[:, :, 0] = -vec_rescaled[i]
    return obs


def args_to_dict(**kwargs):
    """Convert kwargs to a dictionary."""
    return dict(kwargs)


@gin.configurable
def reward_as_dict(**kwargs):
    return args_to_dict(**kwargs)


@gin.configurable
class np_random_seed(object):
    """Run stuff with a fixed random seed."""

    # https://stackoverflow.com/questions/32172054/how-can-i-retrieve-the-current-seed-of-numpys-random-number-generator
    def __init__(self, seed=42):
        self.seed = seed
        self.st0 = None

    def __enter__(self):
        self.st0 = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, type_, value, traceback):
        np.random.set_state(self.st0)


@gin.configurable
def with_fixed_seed(fcn, seed=42, **kwargs):
    """Call function with np.random.seed fixed."""

    with np_random_seed(seed=seed):
        # computing the function
        result = fcn(**kwargs)

    return result

def lstdct2dctlst(lst):
    """List of dictionaries -> dict of lists."""
    keys = lst[0].keys()
    result = {k: [] for k in keys}
    for item in lst:
        for k, v in item.items():
            result[k].append(v)
    return result

def dict_to_sacred(ex, d, iteration, prefix=''):
    """Log a dictionary to sacred."""
    for k, v in d.items():
        if isinstance(v, dict):
            dict_to_sacred(ex, v, iteration, prefix=prefix + k + '/')
        elif isinstance(v, float) or isinstance(v, int):
            ex.log_scalar(prefix + k, v, iteration)