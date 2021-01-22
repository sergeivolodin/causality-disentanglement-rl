import logging
import os

import cloudpickle as pickle
import gin
import numpy as np
import torch
from tqdm import tqdm
from tqdm.auto import tqdm

from causal_util.helpers import postprocess_info
from sparse_causal_model_learner_rl.config import Config
from abc import ABC, abstractmethod, abstractproperty


class AbstractLearner(ABC):
    """Train something."""

    def __init__(self, config, callback=None):
        assert isinstance(config, Config), f"Please supply a valid config: {config}"
        # configuration
        self.config = config

        # call at the end of every epoch with results
        self.callback = callback

        # device to run compute on
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                               and not self.config.get('disable_cuda') else "cpu")
        logging.info(f"Using device {self.device}")

        # map name -> torch model
        self.trainables = {}

        # information from one epoch
        self.epoch_info = None

        # history from epochs
        self.history = []

        # number of epochs trained successfully
        self.epochs = 0

        # how often to checkpoint?
        self.checkpoint_every = self.config.get('checkpoint_every', 10)

        # arguments for trainables
        self.model_kwargs = {}

        # format: [{name}]
        self.potential_trainables_list = self.config.get('potential_trainables_list')
        assert isinstance(self.potential_trainables_list, list),\
            "List of trainables must be a list."

        self._context_cache = None

        # format [[context_var_together_1, ...]]
        self.shuffle_together = []

        # index in the current batch
        self.batch_index = 0

        # shuffle data?
        self.shuffle = self.config.get('shuffle', False)

        # were the trainables already created?
        self.trainables_created = False

    # attributes to save to pickle files
    PICKLE_DIRECTLY = ['history', 'epochs', 'epoch_info', 'config']

    def create_trainables(self):
        """Create the trainables, must be called by the subclass/or on first epoch."""

        if self.trainables_created:
            return
        self.trainables_created = True

        # creating trainables
        for trainable in self.potential_trainables_list:
            cls = self.config.get(trainable['name'], None)
            setattr(self, f"{trainable['name']}_cls", cls)
            if cls:
                logging.info(f"Passing {cls} for {trainable['name']}")
                obj = cls(**self.model_kwargs)
                setattr(self, trainable['name'], obj)
                self.trainables[trainable['name']] = obj
            else:
                logging.warning(f"No class provided for trainable {trainable['name']}")

        self.trainables = {x: y.to(self.device) for x, y in self.trainables.items()}

        def vars_for_trainables(lst):
            """All variables for a list of trainables."""
            return [p for k in lst for p in self.trainables[k].parameters()]

        # All variables for all trainables.
        self.all_variables = vars_for_trainables(self.trainables.keys())

        # filling parameters for optimizers
        self.params_for_optimizers = {
            label: vars_for_trainables(self.config.get('optim_params', {})
                                       .get(label, self.trainables.keys()))
            for label in self.config['optimizers'].keys()}

        # opt_params_descr = {x: [p.name for p in y] for x, y in self.params_for_optimizers.items()}
        # logging.info(f"Optimizers parameters {opt_params_descr}")

        self.optimizer_objects = {}

        for label, fcn in self.config['optimizers'].items():
            params = self.params_for_optimizers[label]
            if params:
                self.optimizer_objects[label] = fcn(params=params)
            else:
                logging.warning(f"No parameters for optimizer {label} {fcn}")

        self._check_execution()

    def checkpoint(self, directory):
        ckpt = os.path.join(directory, "checkpoint")
        with open(ckpt, 'wb') as f:
            pickle.dump(self, f, protocol=2)

        return ckpt

    def train(self, do_tqdm=False):
        """Train (many epochs)."""
        tqdm_ = tqdm if do_tqdm else (lambda x: x)
        for _ in tqdm_(range(self.config['train_steps'])):
            self._epoch()

    @abstractmethod
    def maybe_write_artifacts(self, path_epoch, add_artifact_local):
        """Write artifacts into path_epoch and call add_artifact_local with the path.

        Called by the callback.
        """
        pass

    def __setstate__(self, dct, restore_gin=True):
        # only support gin-defined Configs
        if restore_gin:
            gin.parse_config(dct['gin_config'])
        self.__init__(config=dct['config'])

        # setting attributes
        for key in set(self.PICKLE_DIRECTLY).intersection(dct.keys()):
            setattr(self, key, dct[key])

        new_config = Config()

        for entry in new_config._config.keys():
            if entry not in self.config._config:
                self.config[entry] = new_config._config[entry]
                logging.info("Config entry found in new config but not in old config: " + entry)

        try:
            self.collect_steps()
        except Exception as e:
            logging.error(f"Cannot collect data {e}")
        try:
            self.create_trainables()
        except Exception as e:
            logging.error(f"Cannot create trainables {e}")

        # restoring trainables
        for key in set(self.trainables.keys()).intersection(dct['trainables_weights'].keys()):
            self.trainables[key].load_state_dict(dct['trainables_weights'][key])

    def __getstate__(self):
        result = {k: getattr(self, k) for k in self.PICKLE_DIRECTLY}
        result['trainables_weights'] = {k: v.state_dict() for k, v in self.trainables.items()}
        result['gin_config'] = gin.config_str()
        return result

    @property
    @abstractmethod
    def _context_subclass(self):
        """Datasets/parameters/config for trainables, to be called at every iteration.

        To override."""
        return {}

    @property
    def _context(self):
        context = self._context_subclass
        assert isinstance(context, dict), "_context_subclass() must return a dict."

        context.update({'config': self.config,
                        'trainables': self.trainables,
                        'device': self.device})
        context.update(self.trainables)

        # shuffling groups
        if self.shuffle:
            for group in self.shuffle_together:
                idx = list(range(len(context[group[0]])))
                np.random.shuffle(idx)
                for key in group:
                    context[key] = np.array(context[key])[idx]

        def possible_to_torch(x):
            """Convert a list of inputs into an array suitable for the torch model."""
            if isinstance(x, list) or isinstance(x, np.ndarray):
                x = np.array(x, dtype=np.float32)
                if len(x.shape) == 1:
                    x = x.reshape(-1, 1)
                return torch.from_numpy(x).to(self.device)
            return x

        context = {x: possible_to_torch(y) for x, y in context.items()}
        self._context_cache = context

        return context

    @abstractmethod
    def collect_steps(self):
        """Obtain the dataset and save it internally."""
        pass

    def _epoch(self):
        """One training iteration."""

        # obtain data from environment
        n_batches = collect_every = self.config.get('collect_every', 1)

        if (self.epochs % collect_every == 0) or self._context_cache is None:
            self.collect_steps()
            self.create_trainables()
            context_orig = self._context
            self.batch_index = 0
        else:
            context_orig = self._context_cache

        batch_sizes = []
        if n_batches > 1 and self.config.get('batch_training', False):
            if not self.shuffle:
                logging.warning(f"Shuffle is turned off with n_batches > 1: {n_batches}")

            context = dict(context_orig)

            for group in self.shuffle_together:
                group_len = len(context_orig[group[0]])
                batch_size = group_len // n_batches
                batch_sizes.append(batch_size)
                for item in group:
                    context[item] = context_orig[item][self.batch_index * batch_size:
                                                       (self.batch_index + 1) * batch_size]
                    assert len(context[item]), \
                        f"For some reason, minibatch is empty for {item}{self.epochs} " \
                        f"{self.batch_index} {batch_size} {n_batches} {group_len}. " \
                        f"This should not happen."

            self.batch_index += 1
        else:
            context = context_orig

        epoch_info = {'epochs': self.epochs, 'n_samples': context_orig.get('n_samples', -1),
                      'losses': {},
                      'metrics': {'batch_index': self.batch_index,
                                  'batch_size': np.mean(batch_sizes) if batch_sizes else -1}}

        # train using losses
        for opt_label in sorted(self.optimizer_objects.keys()):
            opt = self.optimizer_objects[opt_label]

            for _ in range(self.config.get('opt_iterations', {}).get(opt_label, 1)):
                opt.zero_grad()
                total_loss = 0
                for loss_label in self.config['execution'][opt_label]:
                    loss = self.config['losses'][loss_label]
                    value = loss['fcn'](**context)
                    if isinstance(value, dict):
                        epoch_info['metrics'].update(value.get('metrics', {}))
                        value = value['loss']
                    coeff = loss['coeff']
                    epoch_info['losses'][f"{opt_label}/{loss_label}/coeff"] = coeff
                    epoch_info['losses'][f"{opt_label}/{loss_label}/value"] = value
                    total_loss += coeff * value
                if hasattr(total_loss, 'backward'):
                    total_loss.backward()
                else:
                    logging.warning(f"Warning: no losses for optimizer {opt_label}")
                epoch_info['losses'][f"{opt_label}/value"] = total_loss
                opt.step()

        if self.epochs % self.config.get('metrics_every', 1) == 0:
            # compute metrics
            for metric_label, metric in self.config['metrics'].items():
                epoch_info['metrics'][metric_label] = metric(**context, context=context,
                                                             prev_epoch_info=self.epoch_info)

        if self.config.get('report_weights', True) and (
                (self.epochs - 1) % self.config.get('report_weights_every', 1) == 0):
            epoch_info['weights'] = {label + '/' + param_name: np.copy(param.detach().cpu().numpy())
                                     for label, trainable in self.trainables.items()
                                     for param_name, param in trainable.named_parameters()}

        # process epoch information
        epoch_info = postprocess_info(epoch_info)

        # send information downstream
        if self.callback:
            self.callback(self, epoch_info)

        if self.config.get('keep_history'):
            self.history.append(epoch_info)

        if self.config.get('max_history_size'):
            mhistsize = int(self.config.get('max_history_size'))
            self.history = self.history[-mhistsize:]

        # update config
        self.config.update(epoch_info=epoch_info)
        self.epochs += 1

        self.epoch_info = epoch_info

        return epoch_info

    def _check_execution(self):
        """Check that all losses are used and all optimizers are used."""
        optimizers_usage = {x: 0 for x in self.config['optimizers']}
        losses_usage = {x: 0 for x in self.config['losses']}
        for opt, losses in self.config['execution'].items():
            optimizers_usage[opt] += 1
            for l in losses:
                losses_usage[l] += 1

        for opt, val in optimizers_usage.items():
            assert val <= 1
            if val == 0:
                logging.warning(f"Warning: optimizer {opt} is unused")

        for loss, val in losses_usage.items():
            if val == 0:
                logging.warning(f"Warning: loss {loss} is unused")
            elif val > 1:
                logging.warning(f"Warning: loss {loss} is used more than once")

    @abstractmethod
    def __repr__(self):
        return f"<AbstractLearner epochs={self.epochs}>"
