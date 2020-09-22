import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
import tensorflow as tf

tf.compat.v1.enable_v2_behavior()
from vectorincrement import VectorIncrementEnvironment
from observation_encoder import KerasEncoder, linear_encoder_unbiased_normal, KerasEncoderWrapper
import numpy as np
import gym
import gin


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


def test_keras_encoder():
    env = gym.make('CartPole-v0')
    gin.enter_interactive_mode()

    def times_two_model(inp_shape, out_shape):
        """Multiply each component by 2."""
        assert inp_shape == out_shape
        assert len(inp_shape) == 1
        model = linear_encoder_unbiased_normal(inp_shape=inp_shape, out_shape=out_shape)
        model.set_weights([np.diag(np.ones(inp_shape[0]) * 2)])
        return model

    gin.bind_parameter("KerasEncoder.model_callable", times_two_model)
    gin.bind_parameter("KerasEncoder.out_shape", env.observation_space.shape)
    gin.bind_parameter("KerasEncoderWrapper.env", env)

    env1 = KerasEncoderWrapper()

    obs_transformed = env1.reset()
    obs_raw = env1.f.raw_observation

    assert np.allclose(obs_raw * 2, obs_transformed)