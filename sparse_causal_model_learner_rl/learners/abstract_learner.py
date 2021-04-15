import logging
import os
import ray

import cloudpickle as pickle
import gin
import numpy as np
import torch
from tqdm import tqdm
from tqdm.auto import tqdm
from copy import deepcopy
from causal_util.helpers import postprocess_info
from sparse_causal_model_learner_rl.config import Config
from abc import ABC, abstractmethod, abstractproperty
from torch.nn.utils.clip_grad import clip_grad_norm_
from sparse_causal_model_learner_rl.loss.helpers import get_loss_and_metrics


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

        # do not convert these to pytorch
        self.no_convert_torch = self.config.get('no_torch', [])

        self.config.maybe_start_communicator()
        self.config.update_communicator()

        self.loss_per_run_cache = {}

    # attributes to save to pickle files
    PICKLE_DIRECTLY = ['history', 'epochs', 'epoch_info', 'config', 'normalizers', 'loss_per_run_cache']


    def create_trainables(self):
        """Create the trainables, must be called by the subclass/or on first epoch."""

        if self.trainables_created:
            return
        self.trainables_created = True

        # creating trainables
        for trainable in self.potential_trainables_list:
            logging.info(f"Potential trainable {trainable['name']}:"
                         f" {trainable.get('description', None)}")
            cls = self.config.get(trainable['name'], None)
            setattr(self, f"{trainable['name']}_cls", cls)
            if cls:
                logging.info(f"Passing {cls} for {trainable['name']}")
                obj = cls(**self.model_kwargs)
                setattr(self, trainable['name'], obj)
                self.trainables[trainable['name']] = obj
                logging.info(f"Created trainable {trainable['name']}")
                logging.info(str(obj.__str__()))
            else:
                logging.warning(f"No class provided for trainable {trainable['name']}")

        self.trainables = {x: y.to(self.device) for x, y in self.trainables.items()}

        def vars_for_trainables(lst):
            """All variables for a list of trainables."""
            result = []
            for k in lst:
                if k.endswith('__params'):
                    k_model, k_attr = k.split('.')
                    k_params = getattr(self.trainables[k_model], k_attr)
                    if callable(k_params):
                        k_params = k_params()
                else:
                    if k in self.trainables and self.trainables[k] is not None:
                        k_params = self.trainables[k].parameters()
                    else:
                        logging.warning(f"Trainable given to optimizer, but not found: {k}")
                        k_params = []
                result += list(k_params)
            return result

        # All variables for all trainables.
        self.all_variables = vars_for_trainables(self.trainables.keys())

        # filling parameters for optimizers
        self.params_for_optimizers = {
            label: vars_for_trainables(self.config.get('optim_params', {})
                                       .get(label, self.trainables.keys()))
            for label in self.config['optimizers'].keys()}

        # opt_params_descr = {x: [p.name for p in y] for x, y in self.params_for_optimizers.items()}
        # logging.info(f"Optimizers parameters {opt_params_descr}")

        self.create_optimizers()

    def create_optimizers(self):

        self.optimizer_objects = {}
        self.scheduler_objects = {}

        for label, fcn in self.config['optimizers'].items():
            params = self.params_for_optimizers[label]
            for p in params:
                if p is None:
                    raise ValueError(f"None param: {label} {fcn} {p}")
            if params:
                opt = fcn(params=params)
                self.optimizer_objects[label] = opt

                sch_cls = self.config.get('schedulers', {}).get(label, None)
                if sch_cls:
                    sch = sch_cls(optimizer=opt)
                    self.scheduler_objects[label] = sch
                    logging.info(f"Creating scheduler for {label}: {sch_cls} {sch}")

            else:
                logging.warning(f"No parameters for optimizer {label} {fcn}")

        self._check_execution()

    def checkpoint(self, directory):
        logging.warning(f"Checkpointing trainer {self}")
        ckpt = os.path.join(directory, "checkpoint")
        with open(ckpt, 'wb') as f:
            pickle.dump(self, f, protocol=2)

        return ckpt

    def train(self, do_tqdm=False):
        """Train (many epochs)."""
        self.create_trainables()
        tqdm_ = tqdm if do_tqdm else (lambda x: x)
        for _ in tqdm_(range(self.epochs, self.config['train_steps'])):
            if self.config.get('detect_anomaly', False):
                with torch.autograd.detect_anomaly():
                    self._epoch()
            else:
                self._epoch()
            self.config.update_communicator()
            f = self.config.get('stopping_condition', None)
            if callable(f) and f(self) is True:
                logging.warning(f"Stopping via the stopping condition")
                break


    @abstractmethod
    def maybe_write_artifacts(self, path_epoch, add_artifact_local):
        """Write artifacts into path_epoch and call add_artifact_local with the path.

        Called by the callback.
        """
        pass

    def __setstate__(self, dct, restore_gin=None):
        if restore_gin is None:
            restore_gin = True

        if Config().get('_unpickle_skip_init', False):
            logging.warning("Not performing initialization, only setting data. The object will not be valid")
            self._unpickled_state = dct
            return

        if Config().get('load_new_config', False):
            logging.warning("Not loading old gin config because using the new config")
            restore_gin = False

        # only support gin-defined Configs
        if restore_gin:
            gin.parse_config(dct['gin_config'])
            config = dct['config']
        else:
            config = Config()

        self.__init__(config=config)

        # setting attributes
        for key in set(self.PICKLE_DIRECTLY).intersection(dct.keys()):
            if not restore_gin and key == 'config':
                logging.warning("Skipping setting old config...")
                continue

            logging.info(f"Loading old {key} from a checkpoint")
            setattr(self, key, dct[key])

        new_config = Config()

        for entry in new_config._config.keys():
            if entry not in self.config._config:
                self.config.set(entry, new_config._config[entry])
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
            logging.info(f"Loading {key} weights from a checkpoint...")
            orig_dict = deepcopy(self.trainables[key].state_dict())
            try:
                self.trainables[key].load_state_dict(dct['trainables_weights'][key], strict=False)
            except RuntimeError as e:
                logging.error(f"Can't load weights for {key}: {e}, undoing loading")
            #    self.trainables[key].load_state_dict(orig_dict)

    def __getstate__(self):
        result = {k: getattr(self, k) for k in self.PICKLE_DIRECTLY
                  if hasattr(self, k)}
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
        return self.wrap_context(context)


    def wrap_context(self, context):
        assert isinstance(context, dict), "_context_subclass() must return a dict."

        context.update({'config': self.config,
                        'trainables': self.trainables,
                        'device': self.device})
        context.update(self.trainables)

        # context transformation
        for fcn in self.config.get('context_transforms', []):
            fcn(self, context)

        # shuffling groups
        if self.shuffle:
            for group in self.shuffle_together:
                lens = [len(context[g_item]) for g_item in group]
                assert all([l == lens[0] for l in lens])
                idx = list(range(len(context[group[0]])))
                np.random.shuffle(idx)
                for key in group:
                    context[key] = np.array(context[key])[idx]

        def possible_to_torch(x, name=""):
            """Convert a list of inputs into an array suitable for the torch model."""

            if name in self.no_convert_torch:
                return x

            try:
                if isinstance(x, list) or isinstance(x, np.ndarray):
                    x = np.array(x, dtype=np.float32)
                    if len(x.shape) == 1:
                        x = x.reshape(-1, 1)
                    return torch.from_numpy(x).to(self.device)
            except Exception as e:
                print(f"Cannot convert {name} to torch representation.")
                raise e
            return x

        context = {x: possible_to_torch(y, name=x) for x, y in context.items()}
        self._context_cache = context

        return context

    def collect_and_get_context(self):
        """Collect new data and return the training context."""
        self.collect_steps()
        return self._context

    @abstractmethod
    def collect_steps(self):
        """Obtain the dataset and save it internally."""
        pass

    def _epoch(self):
        """One training iteration."""

        # obtain data from environment
        n_batches = collect_every = self.config.get('collect_every', 1)

        if (self.epochs % collect_every == 0) or self._context_cache is None:
            # self.collect_steps()
            # self.create_trainables()
            # context_orig = self._context
            context_orig = self.collect_and_get_context()
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
                      'grads': {},
                      'metrics': {'batch_index': self.batch_index,
                                  'batch_size': np.mean(batch_sizes) if batch_sizes else -1}}

        for stats_key in filter(lambda x: x.startswith('context_stats_'), context.keys()):
            epoch_info['metrics'][stats_key] = context[stats_key]

        # train using losses
        loss_epoch_cache = {}
        for opt_label in sorted(self.optimizer_objects.keys()):
            opt = self.optimizer_objects[opt_label]

            # disabling and enabling optimizers
            if self.config.get('opt_enabled_fcn', None):
                fcn = self.config.get('opt_enabled_fcn')
                opt_enabled = fcn(opt_key=opt_label, learner=self)
                if not opt_enabled:
                    continue

            for _ in range(self.config.get('opt_iterations', {}).get(opt_label, 1)):
                opt.zero_grad()
                loss_local_cache = {}
                total_loss = 0
                for loss_label in self.config['execution'].get(opt_label, []):
                    loss = self.config['losses'][loss_label]

                    if loss_label not in self.loss_per_run_cache:
                        self.loss_per_run_cache[loss_label] = {}
                    loss_per_run_cache = self.loss_per_run_cache[loss_label]

                    kwargs = dict(**context, opt_label=opt_label,
                                  loss_local_cache=loss_local_cache,
                                  loss_epoch_cache=loss_epoch_cache,
                                  loss_coeff=loss['coeff'],
                                  loss_per_run_cache=loss_per_run_cache)
                    value, metrics = get_loss_and_metrics(loss['fcn'], **kwargs)
                    epoch_info['metrics'][loss_label] = metrics
                    coeff = loss['coeff']
                    epoch_info['losses'][f"{opt_label}/{loss_label}/coeff"] = coeff
                    epoch_info['losses'][f"{opt_label}/{loss_label}/value"] = value
                    total_loss += coeff * value
                if hasattr(total_loss, 'backward'):
                    total_loss.backward()

                    if self.config.get('grad_clip_value', None) is not None:
                        clip_grad_norm_(self.params_for_optimizers[opt_label],
                                        self.config.get('grad_clip_value'))

                    # computing gradient values
                    grad_norms1 = [torch.mean(torch.abs(p.grad.detach())).item()
                                  for p in self.params_for_optimizers[opt_label]
                                  if p.grad is not None]

                    grad_norms2 = [p.grad.data.norm(2).item() ** 2
                                   for p in self.params_for_optimizers[opt_label]
                                   if p.grad is not None]


                    epoch_info['grads'][f"{opt_label}/grad_total_l1mean"] = np.mean(grad_norms1)
                    epoch_info['grads'][f"{opt_label}/grad_total_l2sum"] = np.sum(grad_norms2) ** 0.5


                else:
                    logging.warning(f"Warning: no losses for optimizer {opt_label}")
                epoch_info['losses'][f"{opt_label}/value"] = total_loss
                opt.step()

                if hasattr(total_loss, 'backward') and opt_label in self.scheduler_objects:
                    self.scheduler_objects[opt_label].step(total_loss)
                    epoch_info['metrics'][f"{opt_label}/scheduler_lr"] = self.scheduler_objects[opt_label]._last_lr[0]

        if self.epochs % self.config.get('metrics_every', 1) == 0:
            # compute metrics
            for metric_label in sorted(self.config['metrics'].keys()):
                metric = self.config['metrics'][metric_label]
                epoch_info['metrics'][metric_label] = metric(**context, context=context,
                                                             learner=self,
                                                             prev_epoch_info=self.epoch_info,
                                                             now_epoch_info=postprocess_info(epoch_info))

        def get_weights():
            return {label + '/' + param_name: np.copy(param.detach().cpu().numpy())
                                     for label, trainable in self.trainables.items()
                                     for param_name, param in trainable.named_parameters()}

        if self.config.get('report_weights', True) and (
                (self.epochs) % self.config.get('report_weights_every', 1) == 0):
            epoch_info['weights'] = get_weights()

        # process epoch information
        epoch_info = postprocess_info(epoch_info)

        # send information downstream
        if self.callback:
            self.callback(self, epoch_info)

        if self.config.get('keep_history'):
            self.history.append(epoch_info)
            if 'weights' not in self.history[-1]:
                self.history[-1]['weights'] = get_weights()
                self.history[-1] = postprocess_info(self.history[-1])

        if self.config.get('max_history_size'):
            mhistsize = int(self.config.get('max_history_size'))
            self.history = self.history[-mhistsize:]

        # update config
        self.config.update(epoch_info=epoch_info, trainables=self.trainables,
                           learner=self, last_context=self._context_cache)
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
