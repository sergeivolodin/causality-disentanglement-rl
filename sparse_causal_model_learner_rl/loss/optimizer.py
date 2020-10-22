import gin
import torch


@gin.configurable
def Optimizer(name, **kwargs):
    return getattr(torch.optim, name)(**kwargs)