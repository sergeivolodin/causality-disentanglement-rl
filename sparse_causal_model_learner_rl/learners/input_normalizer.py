import gin
import numpy as np


@gin.configurable
class Normalizer(object):
    """"Normalize numpy data."""
    def __init__(self, once=True, dim=0, type_='minmax'):
        self.mean = None
        self.std = None

        self.computed = False
        self.type_ = type_

        self.once = once
        self.dim = dim

    def unnormalize(self, outp, eps=1e-8):
        outp = outp.clone()

        if self.type_ == 'minmax':
            outp /= 2 # -0.5, 0.5
            outp += 0.5 # 0, 1
            outp *= (1e-3 + self.max - self.min)
            outp += self.min
            return outp
        elif self.type_ == 'meanstd':
            outp *= (1e-8 + self.std)
            outp += self.mean
            return outp
        else:
            raise NotImplementedError(self.type_)

    def maybe_normalize(self, inp, eps=1e-8):
        if not self.computed or not self.once:
            self.mean = np.mean(inp, axis=self.dim)
            self.std = np.std(inp, axis=self.dim)
            self.min = np.min(inp, axis=self.dim)
            self.max = np.max(inp, axis=self.dim)
            self.computed = True

        if self.type_ == 'meanstd':
            return (inp - self.mean) / (eps + self.std)
        elif self.type_ == 'minmax':
            return 2 * ((inp - self.min) / (1e-3 + self.max - self.min) - 0.5)
        else:
            raise NotImplementedError(f"Wrong type {self.type_}")


@gin.configurable
def normalize_context_transform(self, context, normalize_context_dct=None):
    """Normalize context variables"""

    if normalize_context_dct is None:
        normalize_context_dct = {}

    # cached input normalizer objects
    if not hasattr(self, 'normalizers'):
        self.normalizers = {}

    for norm_with, norm_whiches in normalize_context_dct.items():
        assert norm_with in context, f"To-normalize with item {norm_with} not in context"
        assert isinstance(context[norm_with], np.ndarray), f"To-normalize with item {with_item} is" \
                                                           " not a numpy array"
        assert isinstance(norm_whiches, list), f"norm_whiches must be a list {norm_whiches} for key {norm_with}"

        if norm_with not in self.normalizers:
            self.normalizers[norm_with] = Normalizer()

            # computing the first statistics
            self.normalizers[norm_with].maybe_normalize(context[norm_with])

        for norm_which in norm_whiches:
            assert norm_which in context, f"To-normalize which item {norm_which} not in context"
            assert isinstance(context[norm_which], np.ndarray), f"To-normalize which item {norm_which} is" \
                                                               " not a numpy array"
            context[norm_which] = self.normalizers[norm_with].maybe_normalize(context[norm_which])

    return context
