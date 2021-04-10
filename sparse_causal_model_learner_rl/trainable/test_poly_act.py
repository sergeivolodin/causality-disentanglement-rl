import torch
from torch import nn
import pytest
from sparse_causal_model_learner_rl.trainable.poly_activation import PolyAct


@pytest.mark.parametrize('act_cls,features,batch', [(nn.Tanh, 10, 100), (nn.Tanh, 100, 1000), (nn.Tanh, 5, 5), (nn.ReLU, 10, 100)])
def test_poly_act_out(act_cls, features, batch):
    act = PolyAct(orig_act_cls=act_cls, max_degree=3, features=features)
    inp = torch.randn(batch, features)

    out = act(inp)

    inp_act = act_cls()(inp)
    powers = [torch.ones_like(inp), inp_act, inp_act ** 2, inp_act ** 3]

    out_manual = torch.zeros(batch, features)
    for dim in range(features):
        for power in range(3):
            out_manual[:, dim] += act.a[power, dim] * powers[power][:, dim]

    assert torch.allclose(out, out_manual)
