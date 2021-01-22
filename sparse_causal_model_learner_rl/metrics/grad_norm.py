import gin
import torch
import numpy as np


@gin.configurable
def grad_norm(autoencoder, **kwargs):
    grad = [x.grad for x in autoencoder.parameters()]
    grad = np.max([torch.max(torch.abs(x)).item() for x in grad])
    return grad