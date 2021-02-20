import gym
import gin
from encoder.observation_encoder import linear_encoder_unbiased_normal, KerasEncoderWrapper,\
    ShuffleObservationWrapper
import numpy as np
from .digit_encoder import digit_to_np, digits, small_int_vector_asimage


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

    gin.clear_config()

def test_digits_encoder():
    D = [digit_to_np(i) for i in range(10)]
    for i in range(10):
        for j in range(10):
            if i != j:
                assert np.sum(np.abs(1.0 * D[i] - 1.0 * D[j])) >= 1

    encoded = small_int_vector_asimage([1, 2, 3, 4, 5], max_digits=2)
    assert encoded.shape == (5, 39)

def test_shuffle():
    env = gym.make('CartPole-v0')
    env_shuffle = ShuffleObservationWrapper(env)
    obs_shuffled = env_shuffle.reset()
    obs = env.reset()
    assert obs_shuffled.shape == obs.shape

    # seed is fixed
    assert not np.allclose(obs, obs_shuffled)