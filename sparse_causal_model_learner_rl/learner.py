import argparse

import matplotlib as mpl
mpl.use('Agg')

import sys
import ray
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
import cv2

from causal_util import load_env, WeightRestorer
from causal_util.collect_data import EnvDataCollector, compute_reward_to_go
from sparse_causal_model_learner_rl.config import Config
from sparse_causal_model_learner_rl.trainable.decoder import Decoder
from sparse_causal_model_learner_rl.trainable.model import Model
from sparse_causal_model_learner_rl.trainable.reconstructor import Reconstructor
from sparse_causal_model_learner_rl.trainable.value_predictor import ValuePredictor
from causal_util.helpers import postprocess_info, one_hot_encode
from matplotlib import pyplot as plt
from causal_util.helpers import dict_to_sacred
from sparse_causal_model_learner_rl.sacred_gin_tune.sacred_wrapper import gin_sacred, load_config_files
from causal_util.helpers import lstdct2dctlst
from functools import partial
from sparse_causal_model_learner_rl.visual.learner_visual import total_loss, loss_and_history, plot_contour, plot_3d
import numpy as np
from sparse_causal_model_learner_rl.visual.learner_visual import plot_model, graph_for_matrices, select_threshold
import cloudpickle as pickle
import traceback


class Learner(object):
    """Learn a model for an RL environment with custom losses and parameters."""

    def __init__(self, config, callback=None):
        assert isinstance(config, Config), f"Please supply a valid config: {config}"
        self.config = config
        self.callback = callback

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # creating environment
        self.env = self.create_env()
        self.collector = EnvDataCollector(self.env)

        self.feature_shape = self.config['feature_shape']

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

        self.value_predictor_cls = config.get('value_predictor', None)
        if self.value_predictor_cls:
            assert issubclass(self.value_predictor_cls, ValuePredictor), f"Please supply a valid value predictor class {self.value_predictor_cls}"
            self.value_predictor = self.value_predictor_cls(observation_shape=self.feature_shape)

        # creating a dictionary with all torch models
        self.trainables = {'model': self.model, 'decoder': self.decoder,
                           'reconstructor': self.reconstructor}
        if hasattr(self, 'value_predictor'):
            self.trainables['value_predictor'] = self.value_predictor

        self.history = []
        self.epochs = 0

        print("Using device", self.device)
        self.trainables = {x: y.to(self.device) for x, y in self.trainables.items()}
        self.epoch_info = None

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

        # restoring trainables
        for key in set(self.trainables.keys()).intersection(dct['trainables_weights'].keys()):
            self.trainables[key].load_state_dict(dct['trainables_weights'][key])

    def __getstate__(self):
        result = {k: getattr(self, k) for k in self.PICKLE_DIRECTLY}
        result['trainables_weights'] = {k: v.state_dict() for k, v in self.trainables.items()}
        result['gin_config'] = gin.config_str()
        return result


    def checkpoint(self, directory):
        ckpt = os.path.join(directory, "checkpoint")
        with open(ckpt, 'wb') as f:
            pickle.dump(self, f, protocol=2)

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
        # x: pre time-step, y: post time-step

        # observation
        obs_x = []
        obs_y = []
        obs = []

        # actions
        act_x = []

        # reward-to-go
        reward_to_go = []

        for episode in self.collector.raw_data:
            rew = []
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
                    rew.append(step['reward'])

                if is_multistep and not is_last:
                    obs_x.append(step['observation'])

            rew_to_go_episode = compute_reward_to_go(rew, gamma=self.vf_gamma)
            reward_to_go.extend(rew_to_go_episode)

        # for value function prediction
        assert len(reward_to_go) == len(obs_x)

        # for modelling
        assert len(obs_x) == len(act_x)

        # for reconstruction
        assert len(obs_x) == len(obs_y)

        context = {'obs_x': obs_x, 'obs_y': obs_y, 'action_x': act_x,
                   'obs': obs,
                   'config': self.config,
                   'trainables': self.trainables,
                   'reward_to_go': reward_to_go}
        context.update(self.trainables)

        def possible_to_torch(x):
            """Convert a list of inputs into an array suitable for the torch model."""
            if isinstance(x, list):
                x = np.array(x, dtype=np.float32)
                if len(x.shape) == 1:
                    x = x.reshape(-1, 1)
                return torch.from_numpy(x).to(self.device)
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

        if self.epochs % self.config.get('metrics_every', 1) == 0:
            # compute metrics
            for metric_label, metric in self.config['metrics'].items():
                epoch_info['metrics'][metric_label] = metric(**context, context=context)

        if self.config.get('report_weights', True):
            epoch_info['weights'] = {label + '/' + param_name: np.copy(param.detach().cpu().numpy())
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

        self.epoch_info = epoch_info
        
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
            threshold_act = select_threshold(self.model.Ma, do_plot=do_write, name='learner_action')
            threshold_f = select_threshold(self.model.Mf, do_plot=do_write, name='learner_feature')
            threshold = np.mean([threshold_act, threshold_f])
        ps, f_out = graph_for_matrices(self.model, threshold=threshold, do_write=do_write)
        return threshold, ps, f_out


def main_fcn(config, ex, checkpoint_dir, do_tune=True, do_sacred=True, do_tqdm=False,
             do_exit=True, **kwargs):
    """Main function for gin_sacred."""

    # save graph as artifact

    if do_sacred:
        base_dir = ex.base_dir
    else:
        base_dir = '/tmp/'

    def checkpoint_tune(self, epoch_info=None):
        """Checkpoint, possibly with tune."""
        if epoch_info is None:
            epoch_info = self.epoch_info
        
        if do_tune:
            with tune.checkpoint_dir(step=self.epochs) as checkpoint_dir:
                ckpt = self.checkpoint(checkpoint_dir)
                epoch_info['checkpoint_tune'] = ckpt
                epoch_info['checkpoint_size'] = os.path.getsize(ckpt)
        else:
            ckpt_dir = os.path.join(base_dir, "checkpoint%05d" % epoch_info['epochs'])
            os.makedirs(ckpt_dir, exist_ok=True)
            self.checkpoint(ckpt_dir)
            print(f"Checkpoint available: {ckpt_dir}")
    
    def callback(self, epoch_info):
        """Callback for Learner."""

        epoch_info = dict(epoch_info)

        # chdir to base_dir
        path_epoch = Path(base_dir) / ("epoch%05d" % self.epochs)

        mpl.use('Agg')

        def add_artifact(fn):
            if do_sacred:
                ex.add_artifact(fn, name=("epoch_%05d_" % self.epochs) + os.path.basename(fn))
            else:
                print(f"Artifact available: {fn}")

            # export of images to tensorflow (super slow...)
            if fn.endswith('.png'):
                try:
                    # downscaling the image as ray is slow with big images...
                    img = imread(fn, pilmode='RGB')
                    x, y = img.shape[0:2]
                    factor_x, factor_y = 1, 1
                    mx, my = 150., 150.
                    if x > mx:
                        factor_x = mx / x
                    if y > my:
                        factor_y = my / y

                    factor = min(factor_x, factor_y)

                    if factor != 1:
                        new_shape = (x * factor, y * factor)
                        new_shape = tuple((int(t) for t in new_shape))[::-1]
                        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)

                    img = np.array(img, dtype=np.float32) / 255.

                    img = img.swapaxes(0, 2)
                    img = img.swapaxes(1, 2)
                    # img = np.expand_dims(img, 0)
                    # img = np.expand_dims(img, 0)
                    epoch_info[os.path.basename(fn)[:-4]] = img
                except Exception as e:
                    print(f"Can't read image: {fn} {e} {type(e)}")
                    print(traceback.format_exc())

        # writing figures if requested
        if self.epochs % self.config.get('graph_every', 5) == 0:
            os.makedirs(path_epoch, exist_ok=True)
            with path_epoch:
                try:
                    threshold, ps, f_out = self.visualize_graph(do_write=True)
                    artifact = path_epoch / (f_out + ".png")
                    add_artifact(artifact)
                except Exception as e:
                    print(f"Error plotting causal graph: {self.epochs} {e} {type(e)}")
                    print(traceback.format_exc())

                try:
                    artifact = path_epoch / "threshold_learner_feature.png"
                    add_artifact(artifact)
                except Exception as e:
                    print(f"Error plotting threshold for feature: {self.epochs} {e} {type(e)}")
                    print(traceback.format_exc())

                try:
                    artifact = path_epoch / "threshold_learner_action.png"
                    add_artifact(artifact)
                except Exception as e:
                    print(f"Error plotting threshold for action: {self.epochs} {e} {type(e)}")
                    print(traceback.format_exc())

                try:
                    fig = self.visualize_model()
                    fig.savefig("model.png",  bbox_inches="tight")
                    artifact = path_epoch / "model.png"
                    add_artifact(artifact)
                    plt.clf()
                    plt.close(fig)
                except Exception as e:
                    print(f"Error plotting model: {self.epochs} {e} {type(e)}")
                    print(traceback.format_exc())

        if (self.epochs % self.config.get('loss_every', 100) == 0) and self.history:
            os.makedirs(path_epoch, exist_ok=True)
            with path_epoch:
                try:
                    for opt, (fig, ax) in self.visualize_loss_landscape().items():
                        if self._last_loss_mode == '2d':
                            fig.savefig(f"loss_{opt}.png", bbox_inches="tight")
                            artifact = path_epoch / f"loss_{opt}.png"
                            add_artifact(artifact)
                            plt.clf()
                            plt.close(fig)
                except Exception as e:
                    print(f"Loss landscape error: {type(e)} {str(e)}")
                    print(traceback.format_exc())

        epoch_info['checkpoint_tune'] = None
        if self.epochs % self.checkpoint_every == 0:
            checkpoint_tune(self, epoch_info)

        # pass metrics to sacred
        if self.epochs % self.config.get('report_every', 1) == 0:
            if do_sacred:
                dict_to_sacred(ex, epoch_info, epoch_info['epochs'])
            if do_tune:
                tune.report(**epoch_info)
            if not do_sacred and not do_tune:
                print(f"Report ready, len={len(epoch_info)}")
        else:
            if do_tune:
                tune.report()

    if checkpoint_dir:
        learner = pickle.load(os.path.join(checkpoint_dir, "checkpoint"))
        learner.callback = callback
    else:
        learner = Learner(config, callback=callback)

    learner.train(do_tqdm=do_tqdm)
    
    # last checkpoint at the end
    checkpoint_tune(learner)
    
    # closing all resources
    del learner
    
    if do_exit:
        sys.exit(0)
    
    return None


def learner_gin_sacred(configs, nofail=False):
    """Launch Learner from gin configs."""
    main_fcn_use = main_fcn
    if nofail:
        main_fcn_use = partial(main_fcn, do_exit=False)
    main_fcn_use.__name__ = "main_fcn"
    return gin_sacred(configs, main_fcn_use, db_name='causal_sparse',
                      base_dir=os.path.join(os.getcwd(), 'results'))

parser = argparse.ArgumentParser(description="Causal learning experiment")
parser.add_argument('--config', type=str, required=True, action='append')
parser.add_argument('--n_cpus', type=int, required=False, default=None)
parser.add_argument('--n_gpus', type=int, required=False, default=None)
parser.add_argument('--nowrap', action='store_true')
parser.add_argument('--nofail', help="Disable killing ray actors at the end of the trial", action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()
    cwd = os.getcwd()
    config = args.config
    config = [c if os.path.isabs(c) else os.path.join(cwd, c) for c in config]
    print("Absolute config paths:", config)

    if args.nowrap:
        # useful for debugging/testing
        load_config_files(config)
        config = Config()

        main_fcn(config=config, ex=None, checkpoint_dir=None, do_tune=False, do_sacred=False,
                 do_tqdm=True, do_exit=False)
    else:
        kwargs = {'num_cpus': args.n_cpus}
        if args.n_cpus == 0:
            kwargs = {'num_cpus': 1, 'local_mode': True}
        ray.init(**kwargs, num_gpus=args.n_gpus, include_dashboard=True)
        learner_gin_sacred(config, nofail=args.nofail)
