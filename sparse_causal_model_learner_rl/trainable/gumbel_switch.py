import gin
from torch import nn
import torch


@gin.configurable
class LearnableSwitch(nn.Module):
    """Learn binary probabilistic variables.

    Based on Yoshua Bengio's group work and the Gumbel-Softmax trick.
    """

    def __init__(self, shape):
        super(LearnableSwitch, self).__init__()
        self.shape = shape
        # 1-st component is for ACTIVE
        self.logits = torch.nn.Parameter(torch.ones(2, *shape))

    def softmaxed(self):
        return torch.nn.Softmax(0)(self.logits)[1]

    def sparsify_me(self):
        return [('proba_on', self.softmaxed())]

    def sample_mask(self, method='activate'):

        if method == 'activate':
            return self.softmaxed()
        elif method == 'gumbel':
            return torch.nn.functional.gumbel_softmax(self.logits,
                                                      tau=1, hard=True,
                                                      eps=1e-10, dim=0)[1]
        elif method == 'hard':  # no grad
            return torch.bernoulli(self.softmaxed())
        else:
            raise NotImplementedError

    def forward(self, x):
        return x * self.sample_mask(method='gumbel')


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