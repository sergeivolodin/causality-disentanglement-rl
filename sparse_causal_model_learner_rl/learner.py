import argparse

import matplotlib as mpl
mpl.use('Agg')

import gin
import numpy as np
import torch
from tqdm import tqdm
import uuid
import os
import gym
from ray import tune
from path import Path
from imageio import imread

from causal_util import load_env, WeightRestorer
from causal_util.collect_data import EnvDataCollector
from sparse_causal_model_learner_rl.config import Config
from sparse_causal_model_learner_rl.trainable.decoder import Decoder
from sparse_causal_model_learner_rl.trainable.model import Model
from sparse_causal_model_learner_rl.trainable.reconstructor import Reconstructor
from causal_util.helpers import postprocess_info, one_hot_encode
from matplotlib import pyplot as plt
from causal_util.helpers import dict_to_sacred
from sparse_causal_model_learner_rl.sacred_gin_tune.sacred_wrapper import gin_sacred
from causal_util.helpers import lstdct2dctlst
from functools import partial
from sparse_causal_model_learner_rl.visual.learner_visual import total_loss, loss_and_history, plot_contour, plot_3d
import numpy as np
from sparse_causal_model_learner_rl.visual.learner_visual import plot_model, graph_for_matrices, select_threshold
import cloudpickle as pickle


class Learner(object):
    """Learn a model for an RL environment with custom losses and parameters."""

    def __init__(self, config, callback=None):
        assert isinstance(config, Config), f"Please supply a valid config: {config}"
        self.config = config
        self.callback = callback

        # creating environment
        self.env = self.create_env()
        self.collector = EnvDataCollector(self.env)

        self.feature_shape = self.config['feature_shape']

        self.checkpoint_every = self.config.get('checkpoint_every', 10)

        # Discrete action -> one-hot encoding
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.to_onehot = True
            self.action_shape = (self.env.action_space.n,)
        else:
            self.to_onehot = False
            self.action_shape = self.env.action_space.shape

        self.observation_shape = self.env.observation_space.shape

        # self.action_shape = self.config.get('action_shape')
        # self.observation_shape = self.config.get('observation_shape')

        self.model_cls = config['model']
        assert issubclass(self.model_cls, Model), f"Please supply a valid model class: {self.model_cls}"
        self.model = self.model_cls(feature_shape=self.feature_shape,
                                    action_shape=self.action_shape)

        self.decoder_cls = config['decoder']
        assert self.decoder_cls
        assert issubclass(self.decoder_cls, Decoder), f"Please supply a valid decoder class: {self.decoder_cls}"
        self.decoder = self.decoder_cls(feature_shape=self.feature_shape,
                                        observation_shape=self.observation_shape)

        self.reconstructor_cls = config['reconstructor']
        assert self.reconstructor_cls
        assert issubclass(self.reconstructor_cls,
                          Reconstructor), f"Please supply a valid reconstructor class {self.reconstructor_cls}"
        self.reconstructor = self.reconstructor_cls(feature_shape=self.feature_shape,
                                                    observation_shape=self.observation_shape)

        self.trainables = {'model': self.model, 'decoder': self.decoder,
                           'reconstructor': self.reconstructor}
        self.history = []

        self.epochs = 0

    def checkpoint(self, directory):
        ckpt = os.path.join(directory, "checkpoint")
        with open(ckpt, 'wb') as f:
            # callback contains tune and sacred data and is not pickleable
            old_callback = self.callback
            self.callback = None

            pickle.dump(self, f)

            # restoring callback
            self.callback = old_callback
        return ckpt

    def create_env(self):
        """Create the RL environment."""
        """Create an environment according to config."""
        if 'env_config_file' in self.config:
            gin.parse_config_file(self.config['env_config_file'])
        return load_env()

    def collect_steps(self):
        """Do one round of data collection from the RL environment."""
        # TODO: run a policy with curiosity reward instead of the random policy
        # collecting data
        n_steps = self.config['env_steps']
        while self.collector.steps < n_steps:
            done = False
            self.collector.reset()
            while not done:
                _, _, done, _ = self.collector.step(self.collector.action_space.sample())
        self.collector.flush()

    @property
    def _context(self):
        """Context for losses."""
        obs_x = []
        obs_y = []
        act_x = []
        obs = []

        for episode in self.collector.raw_data:
            is_multistep = len(episode) > 1
            for i, step in enumerate(episode):
                is_first = i == 0
                is_last = i == len(episode) - 1

                obs.append(step['observation'])

                if is_multistep and not is_first:
                    action = step['action']
                    if self.to_onehot:
                        action = one_hot_encode(self.action_shape[0], action)

                    obs_y.append(step['observation'])
                    act_x.append(action)

                if is_multistep and not is_last:
                    obs_x.append(step['observation'])

        context = {'obs_x': obs_x, 'obs_y': obs_y, 'action_x': act_x,
                   'obs': obs,
                   'decoder': self.decoder, 'model': self.model,
                   'reconstructor': self.reconstructor,
                   'config': self.config}

        def possible_to_torch(x):
            """Convert a list of inputs into an array suitable for the torch model."""
            if isinstance(x, list):
                x = np.array(x, dtype=np.float32)
                if len(x.shape) == 1:
                    x = x.reshape(-1, 1)
                return torch.from_numpy(x)
            return x

        context = {x: possible_to_torch(y) for x, y in context.items()}

        return context

    def _epoch(self):
        """One training iteration."""
        # obtain data from environment
        self._check_execution()
        self.collect_steps()

        variables = [p for k in sorted(self.trainables.keys())
                     for p in self.trainables[k].parameters()]

        optimizers = {label: fcn(params=variables)
                      for label, fcn in self.config['optimizers'].items()}

        context = self._context

        epoch_info = {'epochs': self.epochs, 'n_samples': len(context['obs']), 'losses': {},
                      'metrics': {}}

        # train using losses
        for opt_label in sorted(optimizers.keys()):
            opt = optimizers[opt_label]
            opt.zero_grad()
            total_loss = 0
            for loss_label in self.config['execution'][opt_label]:
                loss = self.config['losses'][loss_label]
                value = loss['fcn'](**context)
                coeff = loss['coeff']
                epoch_info['losses'][f"{opt_label}/{loss_label}/coeff"] = coeff
                epoch_info['losses'][f"{opt_label}/{loss_label}/value"] = value
                total_loss += coeff * value
            if hasattr(total_loss, 'backward'):
                total_loss.backward()
            else:
                print(f"Warning: no losses for optimizer {opt_label}")
            epoch_info['losses'][f"{opt_label}/value"] = total_loss
            opt.step()

        # compute metrics
        for metric_label, metric in self.config['metrics'].items():
            epoch_info['metrics'][metric_label] = metric(**context)

        epoch_info['weights'] = {label + '/' + param_name: np.copy(param.detach().numpy())
                                 for label, trainable in self.trainables.items()
                                 for param_name, param in trainable.named_parameters()}

        epoch_info['threshold/action'] = select_threshold(self.model.Ma, do_plot=False)
        epoch_info['threshold/feature'] = select_threshold(self.model.Mf, do_plot=False)

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

        return epoch_info

    @property
    def graph(self):
        """Return the current causal model."""
        return [self.model.Mf, self.model.Ma]

    def train(self, do_tqdm=False):
        """Train (many epochs)."""
        tqdm_ = tqdm if do_tqdm else (lambda x: x)
        for _ in tqdm_(range(self.config['train_steps'])):
            self._epoch()

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
                print(f"Warning: optimizer {opt} is unused")

        for loss, val in losses_usage.items():
            if val == 0:
                print(f"Warning: loss {loss} is unused")
            elif val > 1:
                print(f"Warning: loss {loss} is used more than once")

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
            threshold = select_threshold(self.model.Ma, do_plot=do_write, name='learner')
        ps, f_out = graph_for_matrices(self.model, threshold=threshold, do_write=do_write)
        return threshold, ps, f_out


parser = argparse.ArgumentParser(description="Causal learning experiment")
parser.add_argument('--config', type=str, required=True, action='append')


def main_fcn(config, ex, checkpoint_dir, **kwargs):
    """Main function for gin_sacred."""

    def callback(self, epoch_info):
        """Callback for Learner."""

        epoch_info = dict(epoch_info)

        # save graph as artifact
        uid = str(uuid.uuid4())
        base_dir = ex.base_dir

        # chdir to base_dir
        path_epoch = Path(base_dir) / ("epoch%05d" % self.epochs)

        mpl.use('Agg')

        def add_artifact(fn):
            ex.add_artifact(fn, name=("epoch_%05d_" % self.epochs) + os.path.basename(fn))
            if fn.endswith('.png'):
                try:
                    img = np.array(imread(fn, pilmode='RGB'), dtype=np.float32) / 255.
                    img = img.swapaxes(0, 2)
                    img = img.swapaxes(1, 2)
                    # img = np.expand_dims(img, 0)
                    # img = np.expand_dims(img, 0)
                    epoch_info[os.path.basename(fn)[:-4]] = img
                except Exception as e:
                    print(f"Can't read image: {fn} {e} {type(e)}")

        # writing figures if requested
        if self.epochs % self.config.get('graph_every', 5) == 0:
            os.makedirs(path_epoch, exist_ok=True)
            with path_epoch:
                try:
                    threshold, ps, f_out = self.visualize_graph(do_write=True)
                    artifact = path_epoch / (f_out + ".png")
                    add_artifact(artifact)
                    artifact = path_epoch / "threshold_learner.png"
                    add_artifact(artifact)
                except Exception as e:
                    print(f"Error plotting causal graph: {self.epochs} {e} {type(e)}")

                fig = self.visualize_model()
                fig.savefig("model.png",  bbox_inches="tight")
                artifact = path_epoch / "model.png"
                add_artifact(artifact)

        if (self.epochs % self.config.get('loss_every', 100) == 0) and self.history:
            os.makedirs(path_epoch, exist_ok=True)
            with path_epoch:
                try:
                    for opt, (fig, ax) in self.visualize_loss_landscape().items():
                        if self._last_loss_mode == '2d':
                            fig.savefig(f"loss_{opt}.png", bbox_inches="tight")
                            artifact = path_epoch / f"loss_{opt}.png"
                            add_artifact(artifact)
                except Exception as e:
                    print(f"Loss landscape error: {type(e)} {str(e)}")

        epoch_info['checkpoint_tune'] = None
        if self.epochs % self.checkpoint_every == 0:
            with tune.checkpoint_dir(step=self.epochs) as checkpoint_dir:
                ckpt = self.checkpoint(checkpoint_dir)
                epoch_info['checkpoint_tune'] = ckpt
                epoch_info['checkpoint_size'] = os.path.getsize(ckpt)

        # pass metrics to sacred
        dict_to_sacred(ex, epoch_info, epoch_info['epochs'])
        tune.report(**epoch_info)

    if checkpoint_dir:
        learner = pickle.load(os.path.join(checkpoint_dir, "checkpoint"))
        learner.callback = callback
    else:
        learner = Learner(config, callback=callback)

    learner.train(do_tqdm=False)
    return None


def learner_gin_sacred(configs):
    """Launch Learner from gin configs."""
    return gin_sacred(configs, main_fcn, db_name='causal_sparse',
                      base_dir=os.path.join(os.getcwd(), 'results'))

if __name__ == '__main__':
    import ray
    ray.init(num_cpus=5)
    args = parser.parse_args()
    learner_gin_sacred(args.config)
