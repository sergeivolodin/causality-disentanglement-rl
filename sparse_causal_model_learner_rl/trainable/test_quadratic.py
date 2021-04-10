import torch
import pytest
from sparse_causal_model_learner_rl.trainable.quadratic_neuron import Quadratic


@pytest.mark.parametrize('batch,inf,outf', [(100, 5, 6), (1000, 5, 6), (100, 5, 5), (100, 6, 5), (5, 5, 5)])
def test_quadratic_out(batch, inf, outf):
    q = Quadratic(in_features=inf, out_features=outf)

    q.b.data = torch.randn(q.b.shape)
    q.W.data = torch.randn(q.W.shape)
    q.w.data = torch.randn(q.w.shape)

    inp = torch.randn(batch, inf)
    out = q(inp)

    out_manual = torch.zeros(batch, outf)

    # bias
    out_manual += q.b

    # linear
    out_manual += inp @ q.w.T

    # quadratic
    for b in range(batch):
        for out_dim in range(outf):
            out_manual[b, out_dim] += inp[b, :] @ q.W[out_dim, :, :] @ inp[b, :]

    assert torch.allclose(out_manual, out, atol=1e-5)
