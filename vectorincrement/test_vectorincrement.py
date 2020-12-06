import os

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
import tensorflow as tf

tf.compat.v1.enable_v2_behavior()
from vectorincrement.vectorincrementenv import VectorIncrementEnvironment
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


