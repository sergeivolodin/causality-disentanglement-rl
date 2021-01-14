from .abstract_learner import AbstractLearner
from ..toy_datasets.progressbar import progressbar_image
import numpy as np
from itertools import product
from torch import nn

class ProgressBarLearner(AbstractLearner):
    """Learn on the ProgressBar dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collected = False

    def maybe_write_artifacts(self, path_epoch, add_artifact_local):
        pass

    @property
    def _context_subclass(self):
        pass

    def collect_steps(self):
        if self.collected:
            return
        self.collected = True

        if self.config.get('n_samples') == 'full':
            prod = product([range(t + 1) for t in self.config.get('pb_maxval')])
            X = np.array([progressbar_image(items) for items in prod])
        else:
            raise NotImplementedError

        self.X = X
        self.h, self.w, self.c = X.shape[1:]

    def __repr__(self):
        return f"<ProgressBarLearner>"