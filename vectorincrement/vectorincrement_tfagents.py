import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class VectorIncrementEnvironmentTFAgents(tf_py_environment.TFPyEnvironment):
    """VectorIncrement environment wrapped for TF.agents."""

    def __init__(self, v_n=2, v_k=2, v_seed=2, do_transform=True):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=v_n - 1, name='action')
        self._observation_spec = array_spec.ArraySpec(
            shape=(v_k,), dtype=np.float32, name='observation')
        self._time_step_spec = ts.time_step_spec(self._observation_spec)

        self.env = VectorIncrementEnvironment(n=v_n, k=v_k, seed=v_seed,
                                              do_transform=do_transform)
        self._state = self.env.encoded_state()
        self._episode_ended = False
        self._batched = False

    def batch_size(self):
        return None

    @property
    def batched(self):
        return False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.env.reset()
        self._state = self.env.encoded_state()
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        res = self.env.step(action)
        self._state = self.env.encoded_state()

        return ts.transition(self._state, reward=res['reward'], discount=1.0)