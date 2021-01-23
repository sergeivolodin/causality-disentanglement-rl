import gin
import gym
import numpy as np

from causal_util.helpers import vec_heatmap


@gin.configurable
class LineEnvironment(gym.Env):
    """VectorIncrement environment."""

    def __init__(self, n=3):
        """Initialize.

        Args:
            n: state dimensionality
        """
        self.n = n
        self.observation_space = gym.spaces.Box(low=np.float32(0),
                                                high=np.float32(1), shape=(self.n,))
        self.action_space = gym.spaces.Discrete(2)

        # the state
        self.reset()

        # enabling video recorder
        self.metadata = {'render.modes': ['rgb_array']}

    @property
    def observation(self):
        return np.array(self.s, dtype=np.float32)

    def reset(self):
        """Go back to the start."""
        self.s = np.zeros(self.n)
        self.s[0] = 1
        return self.observation

    def step(self, action):
        """Execute an action."""
        # sanity check
        assert action in [0, 1]

        # past state
        s_old = np.copy(self.s)
        idx = np.argmax(s_old)

        idx_new = idx
        if action == 0 and idx <= self.n - 2: # right
            idx_new += 1
        elif action == 1 and idx >= 1: # left
            idx_new -= 1

        self.s[idx] = 0
        self.s[idx_new] = 1

        obs = self.observation
        rew = np.float32(0)
        done = False
        info = {}

        return obs, rew, done, info

    def __repr__(self):
        return f"Line(n={self.n}, s={self.s})"

    def render(self, mode='rgb_array'):
        return vec_heatmap(self.s, min_value=0, max_value=1)
