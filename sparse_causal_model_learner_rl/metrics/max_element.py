import gin
import numpy as np


@gin.configurable
def nnz(model, eps=1e-3, **kwargs):
    val = 0
    for p in model.parameters():
        val += np.sum(np.abs(p.detach().cpu().numpy()) > eps)
    return val
