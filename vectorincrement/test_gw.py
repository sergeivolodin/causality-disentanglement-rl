from .gridworld import GridWorldNavigationEnv

def test_gw():
    env = GridWorldNavigationEnv(H=2, W=3)
    o = env.reset()
    assert o.shape == (2, 3)

    for episode in range(10):
        o = env.reset()
        assert o.shape == (2, 3)
        for step in range(100):
            o, _, _, _ = env.step(env.action_space.sample())
            assert o.shape == (2, 3)