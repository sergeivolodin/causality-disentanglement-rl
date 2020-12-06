import gin
import numpy as np


@gin.configurable
def max_element_mf(model, **kwargs):
    return np.max(np.abs(model.Mf))

@gin.configurable
def max_element_ma(model, **kwargs):
    return np.max(np.abs(model.Ma))