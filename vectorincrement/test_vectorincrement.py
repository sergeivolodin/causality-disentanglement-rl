import os

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
#import tensorflow as tf

#tf.compat.v1.enable_v2_behavior()
from vectorincrement.vectorincrementenv import VectorIncrementEnvironment
from .sparsematrix import random_sparse_matrix
import gym
import gin
import vectorincrement

import numpy as np
from itertools import product
import pytest

@pytest.mark.parametrize("n", [5, 10, 15])
def test_sparse_matrix_env(n):
    gin.bind_parameter('random_sparse_matrix.n_add_elements_frac', 0.1)
    gin.bind_parameter('SparseMatrixEnvironment.n', n)
    gin.bind_parameter('SparseMatrixEnvironment.matrix_fcn',
                       random_sparse_matrix)
    env = gym.make('SparseMatrix-v0')
    obs0 = env.reset()
    assert obs0.shape == (n,)
    obs1, _, done, _ = env.step(env.action_space.sample())
    assert not done
    assert obs1.shape == (n,)
    obs2, _, done, _ = env.step(env.action_space.sample())
    assert not done
    assert obs2.shape == (n,)
    assert not np.allclose(obs0, obs1)
    assert not np.allclose(obs1, obs2)
    gin.clear_config()

@pytest.mark.parametrize("n,add_frac", product([5, 10, 20],
                                               [0.0, 0.1, 0.2]))
def test_sparse_matrix(n, add_frac):
    A = random_sparse_matrix(n=n, n_add_elements_frac=add_frac)
    
    print(A)
    
    # shape sanity check
    assert A.shape == (n, n)
    
    # non-degeneracy test
    if add_frac == 0:
        assert np.linalg.matrix_rank(A) == n
    
    nonzero_elems_true = n + int(round(add_frac * n * n))
    
    nonzero_elems = np.sum(A != 0)
    assert nonzero_elems == nonzero_elems_true




def test_vectorincrement_steps():
    env = VectorIncrementEnvironment(n=5)

    assert np.allclose(env.reset(), np.zeros(5))

    def step_verify(step_a, step_b):
        """Verify that steps are the same."""
        assert isinstance(step_a, tuple) and isinstance(step_b, tuple), "Wrong type"
        assert len(step_a) == len(step_b), f"Length differ {step_a} {step_b}"
        for val_a, val_b in zip(step_a, step_b):
            assert type(val_a) == type(val_b), f"Types differ {type(val_a)} {type(val_b)} {val_a} {val_b}"
            if isinstance(val_a, np.ndarray):
                assert np.allclose(val_a, val_b), f"Arrays differ {val_a} {val_b}"
            else:
                assert val_a == val_b, f"Values differ {val_a} {val_b}"
        return True

    step_verify(env.step(0), (np.array([1., 0., 0., 0., 0.], dtype=np.float32), np.float32(0.0), False, {}))
    step_verify(env.step(1), (np.array([1., 1., 0., 0., 0.], dtype=np.float32), np.float32(1.0), False, {}))
    step_verify(env.step(0), (np.array([2., 1., 0., 0., 0.], dtype=np.float32), np.float32(0.0), False, {}))
    step_verify(env.step(1), (np.array([2., 2., 0., 0., 0.], dtype=np.float32), np.float32(0.0), False, {}))
    step_verify(env.step(2), (np.array([2., 2., 1., 0., 0.], dtype=np.float32), np.float32(1.0), False, {}))


