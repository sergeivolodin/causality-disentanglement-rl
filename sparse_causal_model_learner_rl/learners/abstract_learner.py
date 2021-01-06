import logging
import os

import cloudpickle as pickle
import torch
from tqdm import tqdm
from tqdm.auto import tqdm

from sparse_causal_model_learner_rl.config import Config


class AbstractLearner(object):
    """Train something."""
    def __init__(self, config, callback=None):
        assert isinstance(config, Config), f"Please supply a valid config: {config}"
        self.config = config
        self.callback = callback

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not self.config.get('disable_cuda') else "cpu")

        # map name -> torch model
        self.trainables = {}

        logging.info(f"Using device {self.device}")
        self.epoch_info = None
        self.history = []
        self.epochs = 0

    def checkpoint(self, directory):
        ckpt = os.path.join(directory, "checkpoint")
        with open(ckpt, 'wb') as f:
            pickle.dumps(self.__getstate__())
            pickle.dump(self, f, protocol=2)

        return ckpt

    def train(self, do_tqdm=False):
        """Train (many epochs)."""
        tqdm_ = tqdm if do_tqdm else (lambda x: x)
        for _ in tqdm_(range(self.config['train_steps'])):
            self._epoch()

    def _epoch(self):
        """Train one epoch."""
        pass

    def maybe_write_artifacts(self, path_epoch, add_artifact_local):
        """Write artifacts into path_epoch and call add_artifact_local with the path."""
        pass