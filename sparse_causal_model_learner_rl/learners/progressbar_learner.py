from .abstract_learner import AbstractLearner
from ..toy_datasets.progressbar import progressbar_image
import numpy as np
from itertools import product
from torch import nn
import logging
from matplotlib import pyplot as plt
import os
import traceback


class ProgressBarLearner(AbstractLearner):
    """Learn on the ProgressBar dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collected = False

    def plot_images_ae(self):
        X = self._context['X']
        idx = np.random.choice(len(X))

        img = X[idx].reshape(self.h, self.w, self.c).detach().cpu().numpy()
        pred = self.autoencoder(X[idx:idx + 1])[0].detach().cpu().numpy()

        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(pred)
        return fig


    def maybe_write_artifacts(self, path_epoch, add_artifact_local):
        if (self.epochs % self.config.get('image_every', 1) == 0):
            os.makedirs(path_epoch, exist_ok=True)
            with path_epoch:
                try:
                    fig = self.plot_images_ae()
                    fig.savefig(f"images.png", bbox_inches="tight")
                    artifact = path_epoch / f"images.png"
                    add_artifact_local(artifact)
                    plt.clf()
                    plt.close(fig)
                except Exception as e:
                    logging.error(f"Images error: {type(e)} {str(e)}")
                    print(traceback.format_exc())

    @property
    def _context_subclass(self):
        return {'X': self.X, 'n_samples': len(self.X)}

    def collect_steps(self):
        if self.collected:
            return
        self.collected = True

        if self.config.get('n_samples') == 'full':
            prod = product(*[range(t + 1) for t in self.config.get('pb_maxval')])
            X = np.array([progressbar_image(values=items) for items in prod])
        else:
            raise NotImplementedError

        self.X = X
        self.h, self.w, self.c = X.shape[1:]

        self.model_kwargs['input_output_shape'] = (self.h, self.w, self.c)

    def __repr__(self):
        return f"<ProgressBarLearner>"