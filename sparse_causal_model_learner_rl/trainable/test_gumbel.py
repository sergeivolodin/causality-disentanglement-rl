from sparse_causal_model_learner_rl.trainable.gumbel_switch import NoiseSwitch
import torch
import numpy as np


def test_noise_switch():
    sw = NoiseSwitch(shape=(5,), noise_add_coeff=1.5, s_init=2.0)
    data_out, sigmas = sw(torch.randn(1000000, 5), return_x_and_mask=True)#, force_proba=0.)
    assert np.abs(data_out.std().item() - (1 ** 2 + (1.5 * 2) ** 2) ** 0.5) < 0.05
    assert np.allclose(sigmas.std().item(), 0)
    assert np.allclose(sigmas.mean().item(), 2.0)
    assert data_out.mean().abs().item() < 0.05
    _, mask = sw(torch.randn(1000000, 5), return_x_and_mask=True, force_proba=1.)
    assert mask.abs().mean().item() < 1e-3
