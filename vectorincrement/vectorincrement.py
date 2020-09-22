import numpy as np
import tensorflow as tf
import gym
import gin


@gin.configurable
class VectorIncrementEnvironment(gym.Env):
    """VectorIncrement environment."""

    def __init__(self, n=10):
        """Initialize.

        Args:
            n: state dimensionality
        """
        self.n = n
        self.observation_space = gym.spaces.Box(low=-np.float32(np.inf),
                                                high=np.float32(np.inf), shape=(self.n,))
        self.action_space = gym.spaces.Discrete(self.n)

        # the state
        self.s = np.zeros(self.n)

    @property
    def observation(self):
        return np.array(self.s, dtype=np.float32)

    def reset(self):
        """Go back to the start."""
        self.s = np.zeros(self.n)
        return self.observation

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

        obs = self.observation
        rew = np.float32(r)
        done = False
        info = {}

        return obs, rew, done, info

    def __repr__(self):
        return f"VectorIncrement(n={self.n}, s={self.s})"

