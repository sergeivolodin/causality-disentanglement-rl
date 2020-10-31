import numpy as np
from sparse_causal_model_learner_rl.trainable.helpers import params_shape, unflatten_params, flatten_params


def test_flat_unflat():
    np.random.seed(42)
    xrand = [np.random.randn(np.random.randint(1, 100), np.random.randint(1, 100)) for _ in
             range(np.random.randint(1, 100))]
    shape = params_shape(xrand)
    assert all([np.allclose(x, y) for x, y in zip(unflatten_params(flatten_params(xrand), shape), xrand)])