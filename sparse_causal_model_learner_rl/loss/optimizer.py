import gin
import torch


@gin.configurable
def Optimizer(name, **kwargs):
    return getattr(torch.optim, name)(**kwargs)

@gin.configurable
def Scheduler(name, **kwargs):
    return getattr(torch.optim.lr_scheduler, name)(**kwargs)