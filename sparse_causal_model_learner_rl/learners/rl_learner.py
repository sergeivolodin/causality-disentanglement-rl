import argparse

from matplotlib import pyplot as plt
import traceback

import matplotlib as mpl

import logging
import ray
import gin
import torch
from tqdm import tqdm
import os
import gym

from causal_util import load_env, WeightRestorer
from causal_util.collect_data import EnvDataCollector, compute_reward_to_go
from sparse_causal_model_learner_rl.config import Config
from sparse_causal_model_learner_rl.trainable.decoder import Decoder
from sparse_causal_model_learner_rl.trainable.model import Model
from sparse_causal_model_learner_rl.trainable.reconstructor import Reconstructor
from sparse_causal_model_learner_rl.trainable.value_predictor import ValuePredictor
from causal_util.helpers import postprocess_info, one_hot_encode
from sparse_causal_model_learner_rl.sacred_gin_tune.sacred_wrapper import load_config_files, main_fcn, learner_gin_sacred
from functools import partial
from sparse_causal_model_learner_rl.visual.learner_visual import total_loss, loss_and_history, plot_contour, plot_3d
import numpy as np
from sparse_causal_model_learner_rl.visual.learner_visual import plot_model, graph_for_matrices, select_threshold
import cloudpickle as pickle
from tqdm.auto import tqdm
from sparse_causal_model_learner_rl.learners.abstract_learner import AbstractLearner


@gin.register
class CausalModelLearnerRL(AbstractLearner):
    """Learn a model for an RL environment with custom losses and parameters."""

    def __init__(self, config, callback=None):

        super().__init__(config, callback)

        # creating environment
        self.env = self.create_env()
        self.collector = EnvDataCollector(self.env)

        self.checkpoint_every = self.config.get('checkpoint_every', 10)
        self.vf_gamma = self.config.get('vf_gamma', 1.0)

        # Discrete action -> one-hot encoding
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.to_onehot = True
            self.action_shape = (self.env.action_space.n,)
        else:
            self.to_onehot = False
            self.action_shape = self.env.action_space.shape

        self.observation_shape = self.env.observation_space.shape

        self.feature_shape = self.config['feature_shape']
        if self.feature_shape is None:
            self.feature_shape = self.observation_shape

        # self.action_shape = self.config.get('action_shape')
        # self.observation_shape = self.config.get('observation_shape')

        # list of potential trainables
        self.potential_trainables = [
            {'name': 'model', 'superclass': Model,
             'kwargs': dict(feature_shape=self.feature_shape, action_shape=self.action_shape)},
            {'name': 'decoder', 'superclass': Decoder,
             'kwargs': dict(feature_shape=self.feature_shape,
                            observation_shape=self.observation_shape)},
            {'name': 'reconstructor', 'superclass': Reconstructor,
             'kwargs': dict(feature_shape=self.feature_shape,
                            observation_shape=self.observation_shape)},
            {'name': 'value_predictor', 'superclass': ValuePredictor,
             'kwargs': dict(observation_shape=self.feature_shape)},
        ]

        # creating trainables
        for trainable in self.potential_trainables:
            cls = config.get(trainable['name'], None)
            setattr(self, f"{trainable['name']}_cls", cls)
            if cls:
                assert issubclass(cls, trainable[
                    'superclass']), f"Please supply a valid {trainable['name']}: {cls}"
                obj = cls(**trainable['kwargs'])
                setattr(self, trainable['name'], obj)
                self.trainables[trainable['name']] = obj
            else:
                logging.warning(f"No class provided for trainable {trainable['name']}")

        self.trainables = {x: y.to(self.device) for x, y in self.trainables.items()}

        def vars_for_trainables(lst):
            return [p for k in lst for p in self.trainables[k].parameters()]

        self.all_variables = vars_for_trainables(self.trainables.keys())

        self.params_for_optimizers = {
            label: vars_for_trainables(self.config.get('optim_params', {}).get(label,
                                                                               self.trainables.keys()))
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

        self._context_cache = None
        self.shuffle_together = []
        self.batch_index = 0
        self.shuffle = self.config.get('shuffle', False)
        self._check_execution()

    # attributes to save to pickle files
    PICKLE_DIRECTLY = ['history', 'epochs', 'epoch_info', 'config']

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
                self.config_config[entry] = new_config._config[entry]
                logging.info("Config entry found in new config but not in old config: " + entry)

        # restoring trainables
        for key in set(self.trainables.keys()).intersection(dct['trainables_weights'].keys()):
            self.trainables[key].load_state_dict(dct['trainables_weights'][key])

    def __getstate__(self):
        result = {k: getattr(self, k) for k in self.PICKLE_DIRECTLY}
        result['trainables_weights'] = {k: v.state_dict() for k, v in self.trainables.items()}
        result['gin_config'] = gin.config_str()
        return result

    def create_env(self):
        """Create the RL environment."""
        """Create an environment according to config."""
        if 'env_config_file' in self.config:
            gin.parse_config_file(self.config['env_config_file'])
        return load_env()

    def collect_steps(self, do_tqdm=False):
        """Do one round of data collection from the RL environment."""
        # TODO: run a policy with curiosity reward instead of the random policy

        # removing old data
        self.collector.clear()

        # collecting data
        n_steps = self.config['env_steps']
        with tqdm(total=n_steps, disable=not do_tqdm) as pbar:
            while self.collector.steps < n_steps:
                done = False
                self.collector.reset()
                pbar.update(1)
                while not done:
                    _, _, done, _ = self.collector.step(self.collector.action_space.sample())
                    pbar.update(1)
        self.collector.flush()

    @property
    def _context(self):
        """Context for losses."""
        # x: pre time-step, y: post time-step

        # observation
        obs_x = []
        obs_y = []
        obs = []

        # actions
        act_x = []

        # reward-to-go
        reward_to_go = []

        episode_rewards = []

        for episode in self.collector.raw_data:
            rew = []
            is_multistep = len(episode) > 1
            for i, step in enumerate(episode):
                remaining_left = i
                remaining_right = len(episode) - 1 - i
                is_first = remaining_left == 0
                is_last = remaining_right == 0

                obs.append(step['observation'])

                if is_multistep and not is_first:
                    action = step['action']
                    if self.to_onehot:
                        action = one_hot_encode(self.action_shape[0], action)

                    obs_y.append(step['observation'])
                    act_x.append(action)
                    rew.append(step['reward'])

                if is_multistep and not is_last:
                    obs_x.append(step['observation'])

            rew_to_go_episode = compute_reward_to_go(rew, gamma=self.vf_gamma)
            episode_rewards.append(rew_to_go_episode[0])
            reward_to_go.extend(rew_to_go_episode)

        # for value function prediction
        assert len(reward_to_go) == len(obs_x)

        # for modelling
        assert len(obs_x) == len(act_x)

        # for reconstruction
        assert len(obs_x) == len(obs_y)

        self.shuffle_together = [['obs_x', 'obs_y', 'action_x', 'reward_to_go'],
                                 ['obs']]

        context = {'obs_x': obs_x, 'obs_y': obs_y, 'action_x': act_x,
                   'obs': obs,
                   'config': self.config,
                   'trainables': self.trainables,
                   'reward_to_go': reward_to_go,
                   'episode_rewards': episode_rewards,
                   'device': self.device}

        # shuffling groups
        if self.shuffle:
            for group in self.shuffle_together:
                idx = list(range(len(context[group[0]])))
                np.random.shuffle(idx)
                for key in group:
                    context[key] = np.array(context[key])[idx]

        context.update(self.trainables)

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

    def _epoch(self):
        """One training iteration."""
        # obtain data from environment
        n_batches = collect_every = self.config.get('collect_every', 1)

        if (self.epochs % collect_every == 0) or self._context_cache is None:
            self.collect_steps()
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
                    context[item] = context_orig[item][self.batch_index * batch_size: (
                                                                                                  self.batch_index + 1) * batch_size]
                    assert len(context[
                                   item]), f"For some reason, minibatch is empty for {item} {self.epochs} {self.batch_index} {batch_size} {n_batches} {group_len}"

            self.batch_index += 1
        else:
            context = context_orig

        epoch_info = {'epochs': self.epochs, 'n_samples': len(context_orig['obs']), 'losses': {},
                      'metrics': {'batch_index': self.batch_index,
                                  'batch_size': np.mean(batch_sizes) if batch_sizes else -1},
                      'episode_reward': np.mean(
                          context['episode_rewards'].detach().cpu().numpy()) if len(
                          context['episode_rewards']) else None}

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

    @property
    def graph(self):
        """Return the current causal model."""
        return [self.model.Mf, self.model.Ma]

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

    def __repr__(self):
        return f"<Learner env={self.env} feature_shape={self.feature_shape} epochs={self.epochs}>"

    def visualize_loss_landscape(self, steps_skip=10, scale=5, n=20, mode='2d'):
        """Plot loss landscape in PCA space with the descent curve."""
        weight_names = [f"{t}/{param}" for t, model in self.trainables.items() for param, _ in
                        model.named_parameters()]

        self._last_loss_mode = mode

        results = {}

        # restore weights to original values
        with WeightRestorer(models=list(self.trainables.values())):
            for opt_label in self.config['optimizers'].keys():
                loss = partial(total_loss, learner=self, opt_label=opt_label)

                loss_w, flat_history = loss_and_history(self, loss, weight_names)
                flat_history = flat_history[::steps_skip]

                if mode == '2d':
                    res = plot_contour(flat_history, loss_w, n=n, scale=scale)
                elif mode == '3d':
                    res = plot_3d(flat_history, loss_w, n=n, scale=scale)
                else:
                    raise ValueError(f"Wrong mode: {mode}, needs to be 2d/3d.")
                results[opt_label] = res

        return results

    def visualize_model(self):
        return plot_model(self.model)

    def visualize_graph(self, threshold='auto', do_write=False):
        if threshold == 'auto':
            _ = select_threshold(self.model.Ma, do_plot=do_write, name='learner_action')
            _ = select_threshold(self.model.Mf, do_plot=do_write, name='learner_feature')
            threshold_act = select_threshold(self.model.Ma, do_plot=False, do_log=False,
                                             name='learner_action')
            threshold_f = select_threshold(self.model.Mf, do_plot=False, do_log=False,
                                           name='learner_feature')
            threshold = np.mean([threshold_act, threshold_f])
        ps, f_out = graph_for_matrices(self.model, threshold_act=threshold_act,
                                       threshold_f=threshold_f, do_write=do_write)
        return threshold, ps, f_out
    
    def maybe_write_artifacts(self, path_epoch, add_artifact_local):
        # writing figures if requested
        if self.epochs % self.config.get('graph_every', 5) == 0:
            os.makedirs(path_epoch, exist_ok=True)
            with path_epoch:
                try:
                    threshold, ps, f_out = self.visualize_graph(do_write=True)
                    artifact = path_epoch / (f_out + ".png")
                    add_artifact_local(artifact)
                except Exception as e:
                    logging.error(f"Error plotting causal graph: {self.epochs} {e} {type(e)}")
                    print(traceback.format_exc())

                try:
                    artifact = path_epoch / "threshold_learner_feature.png"
                    add_artifact_local(artifact)
                except Exception as e:
                    logging.error(
                        f"Error plotting threshold for feature: {self.epochs} {e} {type(e)}")
                    print(traceback.format_exc())

                try:
                    artifact = path_epoch / "threshold_learner_action.png"
                    add_artifact_local(artifact)
                except Exception as e:
                    logging.error(
                        f"Error plotting threshold for action: {self.epochs} {e} {type(e)}")
                    print(traceback.format_exc())

                try:
                    fig = self.visualize_model()
                    fig.savefig("model.png", bbox_inches="tight")
                    artifact = path_epoch / "model.png"
                    add_artifact_local(artifact)
                    plt.clf()
                    plt.close(fig)
                except Exception as e:
                    logging.error(f"Error plotting model: {self.epochs} {e} {type(e)}")
                    print(traceback.format_exc())

        if (self.epochs % self.config.get('loss_every', 100) == 0) and self.history:
            os.makedirs(path_epoch, exist_ok=True)
            with path_epoch:
                try:
                    for opt, (fig, ax) in self.visualize_loss_landscape().items():
                        if self._last_loss_mode == '2d':
                            fig.savefig(f"loss_{opt}.png", bbox_inches="tight")
                            artifact = path_epoch / f"loss_{opt}.png"
                            add_artifact_local(artifact)
                            plt.clf()
                            plt.close(fig)
                except Exception as e:
                    logging.error(f"Loss landscape error: {type(e)} {str(e)}")
                    print(traceback.format_exc())