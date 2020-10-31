from collections import OrderedDict
import numpy as np
import torch


def model_load(model, param_lst):
    """Load parameters into the torch model."""
    model.load_state_dict(OrderedDict(zip(model.state_dict().keys(),
                                          [torch.from_numpy(x) for x in param_lst])))


def flatten_params(params):
    """Flatten parameters of a model."""
    return np.concatenate([x.flatten() for x in params])


def params_shape(params):
    """List of shapes for a list of tensors."""
    return [x.shape for x in params]


def unflatten_params(params_flat, shape):
    """Transform parameters into the original shape from a flat one."""
    assert len(params_flat) == np.sum([np.prod(x) for x in shape])

    out_array = []

    eaten = 0
    for s in shape:
        current_n = np.prod(s)
        current_flat = params_flat[eaten: eaten + current_n]
        current_deflat = current_flat.reshape(s)
        out_array.append(current_deflat)
        eaten += current_n
    return out_array


