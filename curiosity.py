import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tqdm import tqdm

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


def encode_onehot(x, dim):
    """Encode x as 1-hot of dimension d."""
    assert x in range(dim)
    out = np.zeros(dim)
    out[x] = 1
    return out

def buffer_to_dataset(replay_buffer, v_n):
    """Create a dataset from a replay buffer."""
    types = replay_buffer.gather_all().step_type.numpy()[0]
    obs = replay_buffer.gather_all().observation.numpy()[0]
    acts = replay_buffer.gather_all().action.numpy()[0]

    xs = []
    ys = []

    for t, o, a in zip(types, obs, acts):
        oa = np.hstack([o, encode_onehot(a, v_n)])
        if t == 0:
            xs.append(oa)
        elif t == 1:
            xs.append(oa)
            ys.append(o)
        elif t == 2:
            ys.append(o)

    assert len(xs) == len(ys)
    
    return np.array(xs), np.array(ys)

class CuriosityWrapper(wrappers.PyEnvironmentBaseWrapper):
    """Adds a model loss component to the reward."""

    def __init__(self, env, model, alpha=1.0):
        """Initialize.
        
        Args:
            env: tf.agents environment
            model: keras model [observation + one-hot action] -> observation
            alpha: how much l2 norm loss for the model to add to the reward  
        """
        super(CuriosityWrapper, self).__init__(env)

        # saved old time-step
        self.old_step = None

        # keras model taking [obs + one-hot action] and outputting next obs
        self.model = model

        def model_for_obs_and_action(obs, act):
            """Take observation and action as a number, return next observation."""
            v_n = env.wrapped_env().env.n # assuming TimeLimit(VectorIncrementTF)
            z = [np.hstack([obs, encode_onehot(act, v_n)])]
            z = np.array(z, dtype=np.float32)
         #   print(z)
            return self.model(z)
        self.model_for_obs_and_action = model_for_obs_and_action

        self.last_action = None
        
        self.alpha = alpha

    def transform_step(self, step):
        """Replace a reward inside the step to a curiosity reward (r + model loss)"""
        
        # reward to add
        r = 0

        # resetting old step, if required
        if step.step_type == 0:
            self.old_step = None

        # computing the curiosity reward
        if self.old_step is not None:
            # observation predicted by the model
            pred_obs = self.model_for_obs_and_action(
                self.old_step.observation,
                self.last_action)
            
            # computing the reward as l2 norm for the difference
            r = np.linalg.norm(pred_obs - step.observation, ord=1) / np.linalg.norm(step.observation, ord=1)
            #print(pred_obs, step.observation)
        
        # remembering previous step
        self.old_step = step

        # computing the next reward
        total_reward = step.reward + r

        # returning a step with modified reward
        total_reward = np.asarray(total_reward,
                                  dtype=np.asarray(step.reward).dtype)


        return ts.TimeStep(step.step_type, total_reward, step.discount,
                           step.observation)

    def _reset(self):
        return self.transform_step(self._env.reset())

    def _step(self, action):
        # saving the action for transform_step
        self.last_action = action
        return self.transform_step(self._env.step(action))
    
def m_passthrough_action(decoder, v_k, v_n):
    """Create a model with last v_n components in input passed through."""
    inputs = tf.keras.Input(shape=(v_k + v_n,))

    decoded_obs = decoder(inputs[:, :v_k])
    passed_action = inputs[:, v_k:]

    merged = tf.keras.layers.Concatenate()([decoded_obs,  passed_action])

    model = tf.keras.Model(inputs=inputs, outputs=merged)
    
    return model
