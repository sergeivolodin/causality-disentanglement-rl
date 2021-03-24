import gin
import gym
import numpy as np

from causal_util.helpers import vec_heatmap, np_random_seed

class CausalGraphProbas(object):
    """Ground truth model of the environment."""
    def __init__(self, n_actions, n_state, As, Aa):
        assert As.shape == (n_state, n_state)
        assert Aa.shape == (n_state, n_actions)
        self.As = As
        self.Aa = Aa


@gin.configurable
def random_sparse_matrix(n, n_add_elements_frac=None,
                         n_add_elements=None,
                         elements=(-1, 1, -2, 2, 10),
                         add_elements=(-1, 1)):
    """Get a random matrix where there are n_elements."""
    n_total_elements = n * n
    n_diag_elements = n
    frac_diag = 1. * n_diag_elements / n_total_elements
    
    if n_add_elements is not None and n_add_elements_frac is not None:
        raise ValueError("Should only set either n_add_elements or n_add_elements_frac")
        
    if n_add_elements_frac is not None:
        n_add_elements = int(round(n_add_elements_frac * n_total_elements))
    
        assert n_add_elements_frac >= 0, n_add_elements_frac
        assert n_add_elements_frac <= 1 - frac_diag, n_add_elements_frac
    
    assert n_add_elements >= 0
    assert n_add_elements <= n_total_elements - n_diag_elements
    
    A = np.zeros((n, n))
    remaining = set(range(n))
    
    # main elements
    for i in range(n):
        j = np.random.choice(list(remaining))
        remaining.remove(j)
        A[i, j] = np.random.choice(list(elements))
        
    # additional elements
    left_indices = np.array(list(zip(*np.where(A == 0.0))))
#    print(left_indices)
#    print(A)
    np.random.shuffle(left_indices)
    assert len(left_indices) >= n_add_elements
    for i_add in range(n_add_elements):
        i, j = left_indices[i_add]
        assert A[i, j] == 0.0
        A[i, j] = np.random.choice(list(add_elements))
        
    return A


@gin.configurable
class SparseMatrixEnvironment(gym.Env):
    """State is multiplied by a fixed sparse matrix at each time-step."""

    def __init__(self, n=10, seed=42, matrix_fcn=None, use_actions=False):
        """
        Initialize.

        Args:
            n: dimensionality of the state
        """
        self.n = n
        self.use_actions = use_actions
        self.matrix_fcn = matrix_fcn

        # the transition dynamics matrix
        with np_random_seed(seed=seed):
            self.A = matrix_fcn(n=n)

        if self.use_actions:
            with np_random_seed(seed = seed + 1):
                self.Aa = matrix_fcn(n=n)

        self.reset()

        self.observation_space = gym.spaces.Box(low=np.float32(-np.inf),
                                                high=np.float32(np.inf),
                                                shape=(self.n,))
        if self.use_actions:
            self.action_space = gym.spaces.Discrete(self.n)
            n_actions = self.n
            true_Aa = 1. * (np.abs(self.Aa) > 0)
        else:
            self.action_space = gym.spaces.Discrete(1)
            n_actions = 1
            true_Aa = np.zeros((n, 1))
            self.Aa = true_Aa
        self.true_graph = CausalGraphProbas(n_state=n, n_actions=n_actions,
                                            As=1. * (np.abs(self.A) > 0),
                                            Aa=true_Aa)
        
    def render(self, mode='rgb_array'):
        return vec_heatmap(self.state)
        
    def observation(self):
        return np.array(self.state)

    def reset(self):
        # initial state
        self.state = np.random.randn(self.n)
        return self.observation()

    def step(self, action):
        self.state = self.A @ self.state
        if self.use_actions:
            act_infl = self.Aa[:, action]
            act_infl = np.multiply(1. * (self.state > 0), act_infl)
            self.state += act_infl
        obs = self.observation()
        rew = np.float32(0.0)
        done = False
        info = {}

        return obs, rew, done, info
