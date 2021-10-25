import vectorincrement
import gym
import numpy as np


def test_idx_make():
    env = gym.make('IdxEnv-v0')
    assert np.allclose(env.reset()[1:], [1, 0, 0])
    assert np.allclose(env.step(0)[0][1:], [1, 1, 1])
    assert np.allclose(env.step(0)[0][1:], [1, 2, 2])
    assert np.allclose(env.reset()[1:], [2, 0, 2])
    assert np.allclose(env.step(0)[0][1:], [2, 1, 3])
