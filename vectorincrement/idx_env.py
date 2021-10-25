import gin
import gym
import numpy as np
import os

from causal_util.helpers import vec_heatmap


@gin.configurable
class IdxEnv(gym.Env):
    """Increment ID in observations."""

    def __init__(self):
        """Initialize."""
        # format: process id, n_episodes, n_steps, n_total_steps
        self.observation_space = gym.spaces.Box(low=np.float32(0),
                                                high=np.float32('inf'), shape=(4,))
        self.action_space = gym.spaces.Discrete(2)

        # enabling video recorder
        self.metadata = {'render.modes': ['rgb_array']}

        self.pid = os.getpid()
        self.episodes = 0
        self.steps = 0
        self.total_steps = 0

    @property
    def observation(self):
        return np.array([
                   self.pid,
                   self.episodes,
                   self.steps,
                   self.total_steps
               ], dtype=np.float32)

    def reset(self):
        """Go back to the start."""
        self.steps = 0
        self.episodes += 1
        return self.observation

    def step(self, action):
        """Execute an action."""
        self.steps += 1
        self.total_steps += 1

        obs = self.observation
        rew = np.float32(0)
        done = False
        info = {}

        return obs, rew, done, info

    def __repr__(self):
        return f"IdxEnv()"
