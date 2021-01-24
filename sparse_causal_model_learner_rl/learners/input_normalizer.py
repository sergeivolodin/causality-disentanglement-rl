import gin
import numpy as np


@gin.configurable
class Normalizer(object):
    """"Normalize numpy data."""
    def __init__(self, once=True, dim=0):
        self.mean = None
        self.std = None
        self.once = once
        self.dim = dim

    def maybe_normalize(self, inp, eps=1e-8):
        if self.mean is None or not self.once:
            self.mean = np.mean(inp, axis=self.dim)
            self.std = np.std(inp, axis=self.dim)

        return (inp - self.mean) / (eps + self.std)


@gin.configurable
def normalize_context_transform(self, context, normalize_context_lst=None):
    """Normalize context variables"""

    if normalize_context_lst is None:
        normalize_context_lst = []

    # cached input normalizer objects
    if not hasattr(self, 'normalizers'):
        self.normalizers = {}

    for norm_item in normalize_context_lst:
        assert norm_item in context, f"To-normalize item {norm_item} not in context"
        assert isinstance(context[norm_item], np.ndarray), f"To-normalize item {norm_item} is" \
                                                           " not a numpy array"
        if norm_item not in self.normalizers:
            self.normalizers[norm_item] = Normalizer()
        context[norm_item] = self.normalizers[norm_item].maybe_normalize(context[norm_item])

    return context