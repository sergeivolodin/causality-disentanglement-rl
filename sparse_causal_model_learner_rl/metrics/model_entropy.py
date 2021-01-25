import gin
import numpy as np
import logging


def entropy_np(W):
    """Compute entropy for a softmax distribution."""

    # W: shape (n_out, n_in)

    W = W.flatten()

    p_plus = W
    p_minus = 1 - W

    try:
        entropy = np.log(p_plus) * p_plus + p_minus * np.log(p_minus)
        return entropy
    except Exception as e:
        logging.warning(f"Cannot compute entropy for {W}: {e}")
        return -1
        

@gin.configurable
def entropy_action(model, **kwargs):
    return entropy_np(model.Ma)

@gin.configurable
def entropy_features(model, **kwargs):
    return entropy_np(model.Mf)
