from time import time

import gin
import gym
import numpy as np


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