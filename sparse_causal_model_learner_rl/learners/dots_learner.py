from .abstract_learner import AbstractLearner
import numpy as np
import logging
from matplotlib import pyplot as plt
import os
import traceback
from sparse_causal_model_learner_rl.toy_datasets.dots import random_coordinates_n, image_object_positions


class DotsLearner(AbstractLearner):
    """Learn on the Dots dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collected = False

    def plot_images_ae(self, dataset_key='X'):
        X = self._context[dataset_key]
        idx = np.random.choice(len(X))

        img = X[idx].reshape(self.c, self.h, self.w).detach().cpu().numpy()
        pred = self.autoencoder(X[idx:idx + 1])[0].detach().cpu().numpy()

        def chw_hwc(img):
            img = np.swapaxes(img, 1, 2)
            img = np.swapaxes(img, 0, 2)
            return img

        fig = plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(chw_hwc(img))
        plt.subplot(1, 3, 2)
        plt.imshow(chw_hwc(pred))
        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(chw_hwc(img - pred)))
        return fig


    def maybe_write_artifacts(self, path_epoch, add_artifact_local):
        if (self.epochs % self.config.get('image_every', 1) == 0):
            os.makedirs(path_epoch, exist_ok=True)
            with path_epoch:
                for dataset_key in 'X_chw', 'Xtest_chw':
                    try:
                        fig = self.plot_images_ae(dataset_key=dataset_key)
                        fig.savefig(f"images_{dataset_key}.png", bbox_inches="tight")
                        artifact = path_epoch / f"images_{dataset_key}.png"
                        add_artifact_local(artifact)
                        plt.clf()
                        plt.close(fig)
                    except Exception as e:
                        logging.error(f"Images {dataset_key} error: {type(e)} {str(e)}")
                        print(traceback.format_exc())

    @property
    def _context_subclass(self):
        return {'X_chw': self.X_chw, 'n_samples': len(self.X_chw),
                'Xtest_chw': self.Xtest_chw, 'n_test': len(self.Xtest_chw)}

    def collect_steps(self):
        if self.config.get('collect_once', True):
            if self.collected:
                return
            self.collected = True

        coords_fcn = self.config.get('coords_function')

        self.X_hwc = np.array(
            [image_object_positions(positions=coords_fcn())
             for _ in range(self.config.get('n_samples_train'))])
        self.Xtest_hwc = np.array(
            [image_object_positions(positions=coords_fcn())
             for _ in range(self.config.get('n_samples_test'))])

        self.X_chw = np.swapaxes(np.swapaxes(self.X_hwc, 1, 3), 2, 3) # cwh, chw
        self.Xtest_chw = np.swapaxes(np.swapaxes(self.Xtest_hwc, 1, 3), 2, 3)

        self.h, self.w, self.c = self.X_hwc.shape[1:]
        # print("HWC", self.h, self.w, self.c)

        self.model_kwargs['input_output_shape'] = (self.h, self.w, self.c)

        print("COLLECTING DATA!!!")

    def __repr__(self):
        return f"<DotsLearner>"