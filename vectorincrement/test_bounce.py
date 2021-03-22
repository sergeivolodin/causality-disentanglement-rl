from .bounce import BounceEnv

def test_bounce():
    env = BounceEnv()
    o = env.reset()
    assert o.shape == (6,)
    
    assert env.render(resolution=10).shape == (10, 10, 3)

    for episode in range(10):
        o = env.reset()
        assert o.shape == (6,)
        for step in range(10):
            o, _, _, _ = env.step(env.action_space.sample())
            assert o.shape == (6,)