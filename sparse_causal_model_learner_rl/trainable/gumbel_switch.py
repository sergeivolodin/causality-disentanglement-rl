import gin
from torch import nn
import torch
import numpy as np
from functools import partial


@gin.configurable
def sample_from_logits_simple(logits_plus, tau=1.0):
    """Simple sampling, grad=as if multiplied by proba."""
    probas = torch.nn.Sigmoid()(logits_plus)
    probas_tau = torch.nn.Sigmoid()(logits_plus * tau)
    sampled = torch.bernoulli(probas)

    def grad_fcn():
        return probas_tau#1000 * logits_plus

    return sampled + grad_fcn() - grad_fcn().detach()#ogits_plus) - torch.exp(logits_plus).detach()#probas_tau - probas_tau.detach()

@gin.configurable
class LearnableSwitch(nn.Module):
    """Learn binary probabilistic variables.

    Based on Yoshua Bengio's group work and the Gumbel-Softmax trick.
    """

    def __init__(self, shape, sample_many=True,
                 sample_fcn=None,
                 power=1.0, switch_neg=-1,
                 sample_threshold=None,
                 switch_pos=1, tau=1.0):
        super(LearnableSwitch, self).__init__()
        self.shape = shape
        # 1-st component is for ACTIVE

        init_0 = np.ones(shape) * switch_neg
        init_1 = np.ones(shape) * switch_pos
        init = np.array([init_0, init_1])
        #init = np.array(np.ones((2, *shape)), dtype=np.float32)
        init = np.array(init, dtype=np.float32)

        self.logits = torch.nn.Parameter(torch.from_numpy(init))
        self.sample_many = sample_many
        self.power = power
        self.tau = tau
        self.sample_fcn = sample_fcn
        if self.sample_fcn is None:
            self.sample_fcn = partial(torch.nn.functional.gumbel_softmax,
                                      hard=True, eps=1e-10, dim=0)
        self.sample_fcn = partial(self.sample_fcn, tau=self.tau)
        self.sample_threshold = sample_threshold
        if self.sample_threshold is not None:
            def new_sample_fcn(logits, f=self.sample_fcn):
                return self.wrap_sample_threshold(f(logits))
            self.sample_fcn = new_sample_fcn

    def wrap_sample_threshold(self, mask_sampled):
        if self.sample_threshold is None:
            return mask_sampled
        ones = torch.ones_like(mask_sampled)
        out = torch.where(self.softmaxed() > self.sample_threshold,
                          ones, mask_sampled)
        return out.detach() + mask_sampled - mask_sampled.detach()

    def logits_batch(self, n_batch):
        return self.logits.view(self.logits.shape[0], 1,
                                *self.logits.shape[1:]).expand(
            -1, n_batch, *[-1] * (len(self.logits.shape) - 1))

    def softmaxed(self):
        return torch.nn.Softmax(0)(self.logits)[1]

    def sparsify_me(self):
        return [('proba_on', self.softmaxed())]

    def gumbel0(self, logits):
        return self.sample_fcn(logits)[1]

    def sample_mask(self, method='activate'):

        if method == 'activate':
            return self.softmaxed()
        elif method == 'gumbel':
            return self.gumbel0(self.logits)
        elif method == 'hard':  # no grad
            return torch.bernoulli(self.softmaxed())
        else:
            raise NotImplementedError

    def forward(self, x, return_mask=False, return_x_and_mask=False):
        if self.sample_many:
            mask = self.logits_batch(x.shape[0])
            mask = self.gumbel0(mask)
        else:
            mask = self.sample_mask(method='gumbel')

        if return_mask:
            return mask

        # power<1 increases the gradient when the proba is low
        # otherwise, grad ~ proba ** 2 (sampling + here)
        # (xsigma^a')xasigma^a*(1-sigma). the (1-sigma part can still be low)
        # but there the sampling is almost sure
        # loss explodes
        xout = x * mask#.pow(self.power)

        if return_x_and_mask:
            return xout, mask

        return xout

@gin.configurable
class WithInputSwitch(nn.Module):
    """Add the input switch to the model."""

    def __init__(self, model_cls, input_shape, give_mask=False,
                 enable_switch=True, **kwargs):
        super(WithInputSwitch, self).__init__()
        self.switch = LearnableSwitch(shape=input_shape)
        self.give_mask = give_mask

        if give_mask:
            assert len(input_shape) == 1, input_shape
            self.input_dim = self.input_shape[0]
            self.input_shape_with_mask = (2 * self.input_dim,)

            self.model = model_cls(input_shape=self.input_shape_with_mask, **kwargs)
        else:
            self.model = model_cls(input_shape=input_shape, **kwargs)
        self.enable_switch = enable_switch

    def sparsify_me(self):
        return self.switch.sparsify_me()

    def forward(self, x):
        if self.enable_switch:
            on_off, mask = self.switch(x, return_x_and_mask=True)

            if self.give_mask:
                x_with_mask = torch.cat([on_off, mask], dim=1)
                y = self.model(x_with_mask)
            else:
                y = self.model(on_off)
            return y
        else:
            return self.model(x)