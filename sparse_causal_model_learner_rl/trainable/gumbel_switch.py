import gin
from torch import nn
import torch
import numpy as np


@gin.configurable
class LearnableSwitch(nn.Module):
    """Learn binary probabilistic variables.

    Based on Yoshua Bengio's group work and the Gumbel-Softmax trick.
    """

    def __init__(self, shape, sample_many=True):
        super(LearnableSwitch, self).__init__()
        self.shape = shape
        # 1-st component is for ACTIVE

        # init_0 = np.ones(shape) * -1
        # init_1 = np.ones(shape) * 1
        # init = np.array([init_0, init_1])
        init = np.array(np.ones((2, *shape)), dtype=np.float32)

        self.logits = torch.nn.Parameter(torch.from_numpy(init))
        self.sample_many = sample_many

    def logits_batch(self, n_batch):
        return self.logits.view(self.logits.shape[0], 1,
                                *self.logits.shape[1:]).expand(
            -1, n_batch, *[-1] * (len(self.logits.shape) - 1))

    def softmaxed(self):
        return torch.nn.Softmax(0)(self.logits)[1]

    def sparsify_me(self):
        return [('proba_on', self.softmaxed())]

    def gumbel0(self, logits):
        return torch.nn.functional.gumbel_softmax(logits,
                                                  tau=1, hard=True,
                                                  eps=1e-10, dim=0)[1]

    def sample_mask(self, method='activate'):

        if method == 'activate':
            return self.softmaxed()
        elif method == 'gumbel':
            return self.gumbel0(self.logits)
        elif method == 'hard':  # no grad
            return torch.bernoulli(self.softmaxed())
        else:
            raise NotImplementedError

    def forward(self, x, return_mask=False):
        if self.sample_many:
            mask = self.logits_batch(x.shape[0])
            mask = self.gumbel0(mask)
        else:
            mask = self.sample_mask(method='gumbel')

        if return_mask:
            return mask

        return x * mask


@gin.configurable
class WithInputSwitch(nn.Module):
    """Add the input switch to the model."""

    def __init__(self, model_cls, input_shape, **kwargs):
        super(WithInputSwitch, self).__init__()
        self.switch = LearnableSwitch(shape=input_shape)
        self.model = model_cls(input_shape=input_shape, **kwargs)

    def sparsify_me(self):
        return self.switch.sparsify_me()

    def forward(self, x):
        on_off = self.switch(x)
        y = self.model(on_off)
        return y