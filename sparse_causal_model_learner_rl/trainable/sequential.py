import torch
from torch import nn
import gin
from collections import OrderedDict

Conv2d = gin.external_configurable(nn.Conv2d)
Linear = gin.external_configurable(nn.Linear)
OrderedDict = gin.external_configurable(OrderedDict)


@gin.configurable
class Sequential(nn.Module):
    """Sequential torch gin-configurable model."""
    def __init__(self, items, **kwargs):
        super(Sequential, self).__init__()
        assert isinstance(items, list), f"items must be a list {items}"

        self.seq = nn.Sequential(*items)

    def forward(self, x):
        return self.seq(x)


@gin.configurable
class Transpose(nn.Module):
    """Change order of axes."""
    def __init__(self, dim_a, dim_b):
        super(Transpose, self).__init__()
        self.dim_a = dim_a
        self.dim_b = dim_b

    def forward(self, x):
        return torch.transpose(x, self.dim_a, self.dim_b)


@gin.configurable
class ChannelSwap(nn.Module):
    """Change height, width, channels to channels, height, width or vice versa."""
    def __init__(self, order=None):
        super(ChannelSwap, self).__init__()
        self.sw13 = Transpose(1, 3)
        self.sw23 = Transpose(2, 3)
        self.order = order

    def forward(self, x):
        print("CHANNELSWAP", self.order, x.shape)
        if self.order == 'hwc_chw':
            x = self.sw13(x) # cwh
            x = self.sw23(x) # chw
        elif self.order == 'chw_hwc':
            x = self.sw23(x) # cwh
            x = self.sw13(x) # hwc
        else:
            raise NotImplementedError
        return x

@gin.configurable
class Reshape(nn.Module):
    """Reshape the input."""
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape((-1, *self.shape))