from .line import LineEnvironment
import numpy as np

def test_line():
    n = 3
    env = LineEnvironment(n=n)

    obs = env.reset()

    def assert_s(s, obs):
        assert np.argmax(obs) == s
        s_constructed = np.zeros(n)
        s_constructed[s] = 1
        assert np.allclose(s_constructed, obs)

    assert_s(0, obs)

    obs, rew, info, done = env.step(0)

    assert_s(1, obs)

    obs, rew, info, done = env.step(0)

    assert_s(2, obs)

    obs, rew, info, done = env.step(0)

    assert_s(2, obs)

    obs, rew, info, done = env.step(0)

    assert_s(2, obs)

    obs, rew, info, done = env.step(1)

    assert_s(1, obs)

    obs, rew, info, done = env.step(1)

    assert_s(0, obs)

    obs, rew, info, done = env.step(1)

    assert_s(0, obs)