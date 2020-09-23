from observation_encoder import KerasEncoder
import numpy as np
import gin
import gym
from stable_baselines.common.vec_env.base_vec_env import VecEnvWrapper


@gin.configurable
class KerasEncoderVecWrapper(VecEnvWrapper):
    """Apply Keras Encoder to observations, vectorized way."""

    def __init__(self, venv, observation_space=None, action_space=None):
        self._fcn = KerasEncoder(inp_shape=venv.observation_space.shape)
        observation_space = gym.spaces.Box(low=np.float32(-np.inf),
                                           high=np.float32(np.inf),
                                           shape=self._fcn.out_shape)
        super(KerasEncoderVecWrapper, self).__init__(venv, observation_space=observation_space,
                                                     action_space=action_space)
        
    def reset(self):
        obs = self.venv.reset()
        return self._fcn.call_list(obs)
    
    def step_wait(self):
        obs_, rew, done, info = self.venv.step_wait()
        obs = self._fcn.call_list(obs_)
        return obs, rew, done, info
