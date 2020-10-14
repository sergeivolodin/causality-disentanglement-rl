import numpy as np
import torch

from sparse_causal_model_learner_rl.complexity import L1, Lp


def test_L1():
    metric = L1()
    tensor = torch.from_numpy(np.array([1, 2, -1, 0.5], dtype=np.float32))
    val = metric(tensor)
    assert val.numpy() == 4.5


def test_Lp_L1():
    metric = Lp(p=1.0)
    tensor = torch.from_numpy(np.array([1, 2, - 1, 0.5], dtype=np.float32))
    val = metric(tensor)
    assert np.allclose(val.numpy(), 4.5)


def test_Lhalf():
    p = 0.5
    metric = Lp(p=p)
    inp = np.array([1, 2, - 1, 0.5], dtype=np.float32)
    np_val = np.sum(np.abs(inp) ** p) ** (1. / p)
    tensor = torch.from_numpy(inp)
    val = metric(tensor)
    assert np.allclose(val.numpy(), np_val)
