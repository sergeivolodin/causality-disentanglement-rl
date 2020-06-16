import numpy as np
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics, py_metrics
from tf_agents.policies import random_tf_policy, epsilon_greedy_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.networks import actor_distribution_network
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.specs import array_spec
from tf_agents.environments import utils, wrappers
from tf_agents.trajectories import time_step as ts
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.drivers import py_driver

class RandomSophisticatedFunction(object):
    """A function converting an input into a high-dimensional object."""
    def __init__(self, n=10, k=100, seed=11):

        tf.random.set_seed(seed)

        self.model = tf.keras.Sequential([
            #tf.keras.layers.Dense(10, input_shape=(n,), activation='relu'),
            #tf.keras.layers.Dense(100),
            tf.keras.layers.Dense(k, use_bias=False, kernel_initializer='random_normal'),
        ])

    def __call__(self, x):
        return self.model(np.array([x], dtype=np.float32)).numpy()[0]

assert RandomSophisticatedFunction(n=3, k=5, seed=1)([10,10,10]).shape == (5,)

class VectorIncrementEnvironment(object):
    """VectorIncrement environment."""
    def __init__(self, n=10, k=20, do_transform=True, seed=None):
        """Initialize.

        Args:
            n: state dimensionality
            k: observation dimensionality
            do_transform: if False, observation=state, otherwise observation=
              RandomSophisticatedFunction(state) with dimension k
        """
        self.n = n
        self.k = k
        self.e = RandomSophisticatedFunction(n=n, k=k, seed=seed)
        self.s = np.zeros(self.n)
        self.do_transform = do_transform

    def encoded_state(self):
        """Give the current observation."""
        if self.do_transform:
            return np.array(self.e(self.s), dtype=np.float32)
        else:
            return np.array(self.s, dtype=np.float32)  # disabling the encoding completely

    def reset(self):
        """Go back to the start."""
        self.s = np.zeros(self.n)
        return self.encoded_state()

    def step(self, action):
        """Execute an action."""
        # sanity check
        assert action in range(0, self.n)

        # past state
        s_old = np.copy(self.s)

        # incrementing the state variable
        self.s[action] += 1

        # difference between max and min entries
        # of the past state. always >= 0
        maxtomin = max(s_old) - min(s_old)

        # means that there are two different components
        if maxtomin > 0:
            # reward is proportional to the difference between the selected component
            # and the worse component to choose
            r = (max(s_old) - s_old[action]) / maxtomin
        else:
            r = 0

        return {'reward': float(r),
               'state': np.copy(self.s), # new state (do not give to the agent!)
               'observation': self.encoded_state()} # observation (give to the agent)

    def __repr__(self):
        return "VectorIncrement(state=%s, observation=%s)" % (str(self.s), str(self.encoded_state()))

def test_vectorincrement():
    """Test the environment."""
    env = VectorIncrementEnvironment(n=2, k=3, do_transform=False)
    assert np.allclose(env.reset(), np.zeros(2))

    env = VectorIncrementEnvironment(n=2, k=3, seed=42, do_transform=True)
    env.reset()

    result = env.step(0)
    assert result['reward'] == 0
    assert np.allclose(result['state'], np.array([1, 0]))

    assert np.allclose(result['observation'], np.array([0.01637343, -0.04213129,  0.01597168]))

test_vectorincrement()

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

# checking that the environment works
def test_env_transform():
    env = VectorIncrementEnvironmentTFAgents(v_n=10, v_k=50, v_seed=43, do_transform=True)
    env = wrappers.TimeLimit(env, 20)
    utils.validate_py_environment(env, episodes=5)
test_env_transform()

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
def check_env_hardcoded_agent():
    total_reward = hardcoded_agent_reward(2, 2)
    assert total_reward == 10
check_env_hardcoded_agent()
