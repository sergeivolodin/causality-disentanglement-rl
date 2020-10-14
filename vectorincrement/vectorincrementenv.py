import gin
import gym
import numpy as np

from causal_util.helpers import vec_heatmap


@gin.configurable
class SparseMatrixEnvironment(gym.Env):
    """State is multiplied by a fixed sparse matrix at each time-step."""

    def render(self, mode='rgb_array'):
        return vec_heatmap(self.state)

    def __init__(self, n=10):
        """
        Initialize.

        Args:
            n: dimensionality of the state
        """
        self.n = n

        # permutation of 1..n to fill elements in the matrix in
        self.p = np.random.permutation(range(n))

        # the transition dynamics matrix
        self.A = np.zeros((self.n, self.n))

        # filling in the elements
        self.A[range(self.n), self.p] = np.random.choice([-1, 1], size=self.n)

        self.reset()

        self.observation_space = gym.spaces.Box(low=np.float32(-np.inf),
                                                high=np.float32(np.inf), shape=(self.n,))
        self.action_space = gym.spaces.Discrete(1)

    def reset(self):
        # initial state
        self.state = np.random.randn(self.n)
        return np.array(self.state)

    def step(self, action):
        self.state = self.A @ self.state
        obs = np.array(self.state)
        rew = np.float32(0.0)
        done = False
        info = {}

        return obs, rew, done, info


@gin.configurable
class VectorIncrementEnvironment(gym.Env):
    """VectorIncrement environment."""

    def __init__(self, n=10):
        """Initialize.

        Args:
            n: state dimensionality
        """
        self.n = n
        self.observation_space = gym.spaces.Box(low=np.float32(0),
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
