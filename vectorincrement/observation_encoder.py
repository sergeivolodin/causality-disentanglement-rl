import numpy as np
from gym.wrappers import TransformObservation
import tensorflow as tf
import gin
import gym


@gin.configurable
class KerasEncoder(object):
    """Applies a keras model to observations."""

    def __init__(self, model_callable=None, model_filename=None, **kwargs):
        if model_filename is not None:
            print("Loading model", model_filename)
            self.model = tf.keras.models.load_model(model_filename)
        elif model_callable is not None:
            print("Warning: creating a new encoder")
            self.model = model_callable(**kwargs)
        else:
            raise ValueError("Both callable and filename cannot be none")
        self.kwargs = kwargs
        self.last_raw_observation = None
        self.out_shape = self(np.zeros(kwargs['inp_shape'])).shape

    def save(self, fn):
        self.model.save(fn)

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
                       activation='sigmoid', use_bias=True,
                       kernel_initializer='glorot_uniform'):
    """Create a linear keras model."""
    assert len(out_shape) == 1
    if hidden_layers is None:
        hidden_layers = [10, 10]

    layers = [tf.keras.Input(shape=inp_shape)]

    hidden_out_layers = hidden_layers + [out_shape[0]]

    for h in hidden_out_layers:
        layers.append(tf.keras.layers.Dense(h, use_bias=use_bias, activation=activation,
                                            kernel_initializer=kernel_initializer))

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
    def __init__(self, env, **kwargs):
        if isinstance(env, str):
            env = gym.make(env)
        fcn = KerasEncoder(inp_shape=env.observation_space.shape, **kwargs)
        super(KerasEncoderWrapper, self).__init__(env, fcn)
        self.observation_space = gym.spaces.Box(low=np.float32(-np.inf),
                                                high=np.float32(np.inf),
                                                shape=fcn.out_shape)

@gin.configurable
class ObservationScaleWrapper(TransformObservation):
    """Use a keras model to transform observations."""
    def __init__(self, env, scale_coeff=1.0):
        self.scale_coeff = scale_coeff
        def scale_array(x):
            return x * self.scale_coeff
        super(ObservationScaleWrapper, self).__init__(env, scale_array)


def get_obss_states(env, episodes=10):
    """Get observations and original states."""
    assert isinstance(env, KerasEncoderWrapper)
    assert isinstance(episodes, int)
    obss = []
    states = []

    for episode in range(episodes):

        obs = env.reset()
        state = env.f.last_raw_observation

        obss.append(obs)
        states.append(state)

        done = False
        while not done:
            obs, rew, done, info = env.step(env.action_space.sample())
            state = env.f.last_raw_observation

            obss.append(obs)
            states.append(state)

    assert len(obss) == len(states)

    return np.array(obss), np.array(states)

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
