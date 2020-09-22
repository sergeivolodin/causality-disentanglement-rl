import numpy as np
from gym.wrappers import TransformObservation
import tensorflow as tf
import gin


@gin.configurable
class KerasEncoder(object):
    """Applies a keras model to observations."""

    def __init__(self, model_callable, seed=None, **kwargs):
        tf.random.set_seed(seed)
        self.model = model_callable(**kwargs)
        self.kwargs = kwargs
        self.seed = seed
        self.last_raw_observation = None

    def __call__(self, x):
        self.last_raw_observation = x
        return np.float32(self.model(np.array([x], dtype=np.float32)).numpy()[0])

    @property
    def raw_observation(self):
        return self.last_raw_observation

def linear_encoder_unbiased_normal(inp_shape, out_shape):
    """Create a linear keras model."""
    assert len(out_shape) == 1
    model = tf.keras.Sequential([
                tf.keras.layers.Dense(out_shape[0], input_shape=inp_shape,
                                      use_bias=False,
                                      kernel_initializer='random_normal'),
            ])
    return model

@gin.configurable
class KerasEncoderWrapper(TransformObservation):
    """Use a keras model to transform observations."""
    def __init__(self, env):
        fcn = KerasEncoder(inp_shape=env.observation_space.shape)
        super(KerasEncoderWrapper, self).__init__(env, fcn)

# class RandomSophisticatedFunction(object):
#     """A function converting an input into a high-dimensional object."""
#     def __init__(self, n=10, k=100, seed=11):
#
#         tf.random.set_seed(seed)
#
#         self.model = tf.keras.Sequential([
#             #tf.keras.layers.Dense(10, input_shape=(n,), activation='relu'),
#             #tf.keras.layers.Dense(100),
#             tf.keras.layers.Dense(k, use_bias=False, kernel_initializer='random_normal'),
#         ])
#
#     def __call__(self, x):
#         return self.model(np.array([x], dtype=np.float32)).numpy()[0]
#
# assert RandomSophisticatedFunction(n=3, k=5, seed=1)([10,10,10]).shape == (5,)