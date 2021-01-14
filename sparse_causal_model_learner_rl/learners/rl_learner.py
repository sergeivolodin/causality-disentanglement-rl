import logging
import os
import traceback
from functools import partial

import gin
import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from tqdm.auto import tqdm

from causal_util import load_env, WeightRestorer
from causal_util.collect_data import EnvDataCollector, compute_reward_to_go
from causal_util.helpers import one_hot_encode
from sparse_causal_model_learner_rl.learners.abstract_learner import AbstractLearner
from sparse_causal_model_learner_rl.visual.learner_visual import plot_model, graph_for_matrices, \
    select_threshold
from sparse_causal_model_learner_rl.visual.learner_visual import total_loss, loss_and_history, \
    plot_contour, plot_3d


@gin.register
class CausalModelLearnerRL(AbstractLearner):
    """Learn a model for an RL environment with custom losses and parameters."""

    def __init__(self, config, callback=None):

        super().__init__(config, callback)

        # creating environment
        self.env = self.create_env()
        self.collector = EnvDataCollector(self.env)

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

        self.model_kwargs = {'feature_shape': self.feature_shape,
                             'action_shape': self.action_shape,
                             'observation_shape': self.observation_shape}

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
    def _context_subclass(self):
        """Context for losses."""
        # x: pre time-step, y: post time-step

        # observations, actions, rewards-to-go, total rewards
        obs_x, obs_y, obs, act_x, reward_to_go, episode_rewards = [], [], [], [], []

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
                   'reward_to_go': reward_to_go,
                   'episode_rewards': episode_rewards,
                   'n_samples': len(obs_x)}

        return context

    @property
    def graph(self):
        """Return the current causal model."""
        return [self.model.Mf, self.model.Ma]

    def __repr__(self):
        return f"<RLLearner env={self.env} feature_shape={self.feature_shape} " \
               f"epochs={self.epochs}>"

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
