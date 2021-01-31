import gin
import numpy as np
import logging


@gin.configurable
def entropy_np(W, return_distribution=False, eps=1e-8):
    """Compute entropy for a softmax distribution."""

    # W: shape (n_out, n_in)

    W = W.flatten()
    
    W = np.clip(W, a_min=eps, a_max=1.0)

    p_plus = W
    p_minus = 1 - W

    try:
        entropy = np.log(p_plus) * p_plus + p_minus * np.log(p_minus)
        entropy = -entropy
        if return_distribution:
            return entropy
        else:
            return np.mean(entropy)
    except Exception as e:
        logging.warning(f"Cannot compute entropy for {W}: {e}")
        return -1
        

@gin.configurable
def entropy_action(model, **kwargs):
    return entropy_np(model.Ma)

@gin.configurable
def entropy_features(model, **kwargs):
    return entropy_np(model.Mf)
