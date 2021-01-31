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

class Switch(nn.Module):
    def __init__(self, shape, sample_many=True):
        super(Switch, self).__init__()
        self.shape = shape
        self.sample_many = sample_many

    def sample_mask(self, method=None):
        raise NotImplementedError

    def logits_batch(self, n_batch):
        raise NotImplementedError

    def gumbel0(self, data):
        raise NotImplementedError

    def softmaxed(self):
        raise NotImplementedError

    def sparsify_me(self):
        raise NotImplementedError

    def project(self):
        pass

    def forward(self, x, return_mask=False, return_x_and_mask=False):
        self.project()
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
        # print(x.shape, mask.shape, self.probas.shape)
        xout = x * mask#.pow(self.power)

        if return_x_and_mask:
            return xout, mask

        return xout

@gin.configurable
class LearnableSwitchSimple(Switch):
    """Sample from Bernoulli, return p for grad."""
    def __init__(self, initial_proba=0.5, **kwargs):
        super(LearnableSwitchSimple, self).__init__(**kwargs)

        init = np.full(shape=self.shape, fill_value=initial_proba, dtype=np.float32)
        self.probas = torch.nn.Parameter(torch.from_numpy(init))

    def logits_batch(self, n_batch):
        return self.probas.view(1, *self.probas.shape).expand(
            n_batch, *[-1] * (len(self.probas.shape)))

    def project(self):
        self.probas.data = torch.clamp(self.probas.data, min=0.0, max=1.0)

    def gumbel0(self, data):
        sampled = torch.bernoulli(data)
        return sampled + data - data.detach()

    def sample_mask(self, method=None):
        return self.gumbel0(self.probas)

    def softmaxed(self):
        return self.probas

    def sparsify_me(self):
        return [('proba_on', self.softmaxed())]


@gin.configurable
class LearnableSwitch(Switch):
    """Learn binary probabilistic variables.

    Based on Yoshua Bengio's group work and the Gumbel-Softmax trick.
    """

    def __init__(self, sample_fcn=None,
                 power=1.0, switch_neg=-1,
                 sample_threshold=None,
                 sample_threshold_min=None,
                 switch_pos=1, tau=1.0, **kwargs):
        super(LearnableSwitch, self).__init__(**kwargs)
        # 1-st component is for ACTIVE

        init_0 = np.ones(self.shape) * switch_neg
        init_1 = np.ones(self.shape) * switch_pos
        init = np.array([init_0, init_1])
        #init = np.array(np.ones((2, *shape)), dtype=np.float32)
        init = np.array(init, dtype=np.float32)

        self.logits = torch.nn.Parameter(torch.from_numpy(init))
        self.power = power
        self.tau = tau
        self.sample_fcn = sample_fcn
        if self.sample_fcn is None:
            self.sample_fcn = partial(torch.nn.functional.gumbel_softmax,
                                      hard=True, eps=1e-10, dim=0)
        self.sample_fcn = partial(self.sample_fcn, tau=self.tau)
        self.sample_threshold = sample_threshold
        self.sample_threshold_min = sample_threshold_min
        if self.sample_threshold is not None:
            def new_sample_fcn(logits, f=self.sample_fcn):
                return self.wrap_sample_threshold(f(logits))
            self.sample_fcn = new_sample_fcn
        if self.sample_threshold_min is not None:
            def new_sample_fcn(logits, f=self.sample_fcn):
                return self.wrap_sample_threshold_min(f(logits))
            self.sample_fcn = new_sample_fcn

    def wrap_sample_threshold_min(self, mask_sampled):
        if self.sample_threshold_min is None:
            return mask_sampled
        zeros = torch.zeros_like(mask_sampled)
        out = torch.where(self.softmaxed() < self.sample_threshold_min,
                          zeros, mask_sampled)
        return out.detach() + mask_sampled - mask_sampled.detach()

            
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

@gin.configurable
class WithInputSwitch(nn.Module):
    """Add the input switch to the model."""

    def __init__(self, model_cls, input_shape, give_mask=False,
                 switch_cls=LearnableSwitch,
                 enable_switch=True, **kwargs):
        super(WithInputSwitch, self).__init__()
        self.n_models = kwargs.get('n_models', None)
        if self.n_models is not None:
            self.input_shape = (*input_shape, self.n_models)
        else:
            self.input_shape = input_shape
        self.switch = switch_cls(shape=self.input_shape)
        self.give_mask = give_mask

        if give_mask:
            assert len(input_shape) == 1, input_shape
            self.input_dim = input_shape[0]
            self.input_shape_with_mask = (2 * self.input_dim,)

            self.model = model_cls(input_shape=self.input_shape_with_mask, **kwargs)
        else:
            self.model = model_cls(input_shape=input_shape, **kwargs)
        self.enable_switch = enable_switch

    def sparsify_me(self):
        return self.switch.sparsify_me()

    def forward(self, x, enable_switch=None, detach_mask=False):

        if self.n_models is not None:
            x = x.view(*x.shape, 1).expand(*[-1] * len(x.shape), self.n_models)

        if enable_switch is None:
            enable_switch = self.enable_switch
        if enable_switch:

            # print("XSHAPE", x.shape)

            on_off, mask = self.switch(x, return_x_and_mask=True)

            if self.give_mask:
                mask_maybe_detached = mask.detach() if detach_mask else mask
                x_with_mask = torch.cat([on_off, mask_maybe_detached], dim=1)
                y = self.model(x_with_mask)
            else:
                y = self.model(on_off)
            return y
        else:
            if self.give_mask:
                ones = torch.ones_like(x)
                x_with_mask = torch.cat([x, ones], dim=1)
                y = self.model(x_with_mask)
            else:
                y = self.model(x)
            return y