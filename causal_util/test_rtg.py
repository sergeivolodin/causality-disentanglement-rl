from causal_util.collect_data import compute_reward_to_go
import numpy as np

def test_rtg_single():
    assert np.allclose(np.array(compute_reward_to_go([1], 0.5)), np.array([1]))

def test_rtg_3(gamma=0.4):
    src = np.array([1, 5, 4], dtype=np.float32)
    res = np.array(compute_reward_to_go(src, gamma))
    res_manual = np.zeros(3)
    res_manual[0] = src[0] + src[1] * gamma + src[2] * gamma ** 2
    res_manual[1] = src[1] + src[2] * gamma
    res_manual[2] = src[2]
    assert np.allclose(res, res_manual), (res, res_manual)