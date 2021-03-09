import gin
import numpy as np
from causal_util import get_true_graph
from scipy.spatial.distance import cosine


@gin.configurable
def cosine_to_true(model, learner, **kwargs):
    if learner is None:
        return -1.
    
    if not hasattr(learner, 'env'):
        return -1.
    
    g = get_true_graph(learner.env)
    
    if g is None:
        return -2.
    
    Mf = model.Mf
    Af = g.As
    
    if Mf.shape != Af.shape:
        return -3.
    
    return cosine(Mf.flatten(), Af.flatten())