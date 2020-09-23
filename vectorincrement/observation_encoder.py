import numpy as np
from gym.wrappers import TransformObservation
import tensorflow as tf
import gin
import gym


@gin.configurable
class KerasEncoder(object):
    """Applies a keras model to observations."""

    def __init__(self, model_callable, seed=None, **kwargs):
        if hasattr(tf.random, 'set_seed'):
            tf.random.set_seed(seed)
        else:
            tf.set_random_seed(seed)
        self.model = model_callable(**kwargs)
        self.kwargs = kwargs
        self.seed = seed
        self.last_raw_observation = None
        self.out_shape = self(np.zeros(kwargs['inp_shape'])).shape

    def __call__(self, x):
        self.last_raw_observation = x
        return np.float32(self.model.predict(np.array([x], dtype=np.float32))[0])

    def call_list(self, x):
        assert isinstance(x, list) or isinstance(x, np.ndarray)
        self.last_raw_observations = x
        return list(np.float32(self.model.predict(np.array(x, dtype=np.float32))))

    @property
    def raw_observation(self):
        return self.last_raw_observation


@gin.configurable
def non_linear_encoder(inp_shape, out_shape, hidden_layers=None,
                       activation='sigmoid', use_bias=True):
    """Create a linear keras model."""
    assert len(out_shape) == 1
    if hidden_layers is None:
        hidden_layers = [10, 10]

    layers = [tf.keras.Input(shape=inp_shape)]

    hidden_out_layers = hidden_layers + [out_shape[0]]

    for h in hidden_out_layers:
        layers.append(tf.keras.layers.Dense(h, use_bias=use_bias, activation=activation,
                                            kernel_initializer='random_normal'))

    model = tf.keras.Sequential(layers)
    return model

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
        if isinstance(env, str):
            env = gym.make(env)
        fcn = KerasEncoder(inp_shape=env.observation_space.shape)
        super(KerasEncoderWrapper, self).__init__(env, fcn)
        self.observation_space = gym.spaces.Box(low=np.float32(-np.inf),
                                                high=np.float32(np.inf),
                                                shape=fcn.out_shape)

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
