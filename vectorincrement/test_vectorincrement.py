import tensorflow as tf

tf.compat.v1.enable_v2_behavior()
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils, wrappers
from vectorincrement import VectorIncrementEnvironment
import numpy as np


def test_vectorincrement_steps():
    env = VectorIncrementEnvironment(n=5)

    assert np.allclose(env.reset(), np.zeros(5))

    def step_verify(step_a, step_b):
        """Verify that steps are the same."""
        assert isinstance(step_a, tuple) and isinstance(step_b, tuple), "Wrong type"
        assert len(step_a) == len(step_b), f"Length differ {step_a} {step_b}"
        for val_a, val_b in zip(step_a, step_b):
            assert type(val_a) == type(val_b), f"Types differ {type(val_a)} {type(val_b)} {val_a} {val_b}"
            if isinstance(val_a, np.ndarray):
                assert np.allclose(val_a, val_b), f"Arrays differ {val_a} {val_b}"
            else:
                assert val_a == val_b, f"Values differ {val_a} {val_b}"
        return True

    step_verify(env.step(0), (np.array([1., 0., 0., 0., 0.], dtype=np.float32), np.float32(0.0), False, {}))
    step_verify(env.step(1), (np.array([1., 1., 0., 0., 0.], dtype=np.float32), np.float32(1.0), False, {}))
    step_verify(env.step(0), (np.array([2., 1., 0., 0., 0.], dtype=np.float32), np.float32(0.0), False, {}))
    step_verify(env.step(1), (np.array([2., 2., 0., 0., 0.], dtype=np.float32), np.float32(0.5), False, {}))
    step_verify(env.step(2), (np.array([2., 2., 1., 0., 0.], dtype=np.float32), np.float32(1.0), False, {}))


# checking that the environment works
def test_env_transform():
    env = VectorIncrementEnvironmentTFAgents(v_n=10, v_k=50, v_seed=43, do_transform=True)
    env = wrappers.TimeLimit(env, 20)
    utils.validate_py_environment(env, episodes=5)


def hardcoded_agent_reward(v_n, v_k, time_limit=20):
    env = VectorIncrementEnvironmentTFAgents(v_n=v_n, v_k=v_k, do_transform=False)
    env = wrappers.TimeLimit(env, 20)
    train_env = tf_py_environment.TFPyEnvironment(env)

    # running a hardcoded agent to test if the environment works correctly
    o = train_env.reset().observation.numpy()[0]
    total_reward = 0
    while True:
        act = np.argmin(o)
        step = train_env.step(act)
        o = step.observation.numpy()[0]
        r = np.array(step.reward[0])
        total_reward += r
        if step.step_type == 2:
            return total_reward


# checking that the environment works
def test_env_hardcoded_agent():
    total_reward = hardcoded_agent_reward(2, 2)
    assert total_reward == 10
