import logging
import os
import traceback
from functools import partial

import gin
import numpy as np
import ray
from matplotlib import pyplot as plt

from causal_util import WeightRestorer
from sparse_causal_model_learner_rl.learners.abstract_learner import AbstractLearner
from sparse_causal_model_learner_rl.visual.learner_visual import plot_model, graph_for_matrices, \
    select_threshold
from sparse_causal_model_learner_rl.visual.learner_visual import total_loss, loss_and_history, \
    plot_contour, plot_3d
from .rl_data import RLContext, ParallelContextCollector, get_shuffle_together
from sparse_causal_model_learner_rl.metrics import find_value


@gin.configurable
def causal_learner_stopping_condition(learner, edges_Mf=None, edges_Ma=None, metric_geq=None,
                                      graph_threshold=0.01):
    """Stop if sparsity <= z and losses <= c."""
    Mf, Ma = learner.graph
    info = learner.epoch_info
    nnz_Mf = np.sum(Mf > graph_threshold)
    nnz_Ma = np.sum(Ma > graph_threshold)
    
    if edges_Mf is not None and nnz_Mf > edges_Mf:
        return False
    if edges_Ma is not None and nnz_Ma > edges_Ma:
        return False
    for metric_key, metric_thr in metric_geq.items():
        val = find_value(info, metric_key)
        if val < metric_thr:
            return False
    print("Feature graph matrix", Mf)
    print("Action graph matrix", Ma)
    return True

@gin.register
class CausalModelLearnerRL(AbstractLearner):
    """Learn a model for an RL environment with custom losses and parameters."""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # creating environment
        self.rl_context = RLContext(config=self.config)
        self.env = self.rl_context.env

        self.observation_shape = self.env.observation_space.shape
        self.action_shape = self.rl_context.action_shape

        self.feature_shape = self.config['feature_shape']
        if self.feature_shape is None:
            self.feature_shape = self.observation_shape

        self.additional_feature_keys = self.rl_context.additional_feature_keys
        self.additional_feature_shape = (len(self.additional_feature_keys),)

        self.model_kwargs = {'feature_shape': self.feature_shape,
                             'action_shape': self.action_shape,
                             'observation_shape': self.observation_shape,
                             'additional_feature_shape': self.additional_feature_shape,
                             'total_feature_shape': (self.feature_shape[0] +
                                                     self.additional_feature_shape[0],)}

        logging.info(self.model_kwargs)

        self.collect_remotely = self.config.get('collect_remotely', False)
        if self.collect_remotely:
            self.remote_rl_context = ParallelContextCollector(config=self.config)
        self.shuffle_together = get_shuffle_together(self.config)
        self.run_normalizers_at_start()

    def collect_steps(self):
        raise NotImplementedError("Use collect_and_get_context")

    @property
    def _context_subclass(self):
        raise NotImplementedError("Use collect_and_get_context")

    def context_add_scalars(self, ctx):
        ctx['n_samples'] = len(ctx['obs_x'])
        ctx['additional_feature_keys'] = self.additional_feature_keys
        return ctx

    def run_normalizers_at_start(self):
        if self.collect_remotely:
            self.collect_and_get_context(full_buffer_now=True)
            if self.collect_remotely:
                self.remote_rl_context.collect_initial()
        else:
            logging.warning("Dataset size is small for normalizers in non-parallel mode")


    def collect_and_get_context(self, full_buffer_now=False):
        """Collect new data and return the training context."""

        if self.collect_remotely:
            if full_buffer_now:
                pre_context = self.remote_rl_context.get_whole_buffer()
            else:
                pre_context = self.remote_rl_context.collect_get_context()
        else:
            if full_buffer_now:
                raise NotImplementedError("Full buffer only supported with remote context...")
            else:
                self.rl_context.collect_steps()
                pre_context = self.rl_context.get_context()

        pre_context_sample = pre_context
        pre_context_sample = self.context_add_scalars(pre_context_sample)
        return self.wrap_context(pre_context_sample)

    @property
    def graph(self):
        """Return the current causal model."""
        return [self.model.Mf, self.model.Ma]

    def __repr__(self):
        return f"<RLLearner env={self.env} feature_shape={self.feature_shape} " \
               f"epochs={self.epochs} additional_feature_shape={self.additional_feature_shape}>"

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

    def visualize_model(self, model=None):
        if model is None:
            model = self.model
        return plot_model(model)

    def visualize_graph(self, threshold='auto', do_write=False, model=None):
        if model is None:
            model = self.model
        if threshold == 'auto':
            _ = select_threshold(model.Ma, do_plot=do_write, name='learner_action')
            _ = select_threshold(model.Mf, do_plot=do_write, name='learner_feature')
            threshold_act = select_threshold(model.Ma, do_plot=False, do_log=False,
                                             name='learner_action')
            threshold_f = select_threshold(model.Mf, do_plot=False, do_log=False,
                                           name='learner_feature')
            threshold = np.mean([threshold_act, threshold_f])
        ps, f_out = graph_for_matrices(model, threshold_act=threshold_act,
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
