import argparse

import gin
import numpy as np
import torch
from tqdm import tqdm
import uuid
import pickle
import os

from causal_util import load_env
from causal_util.collect_data import EnvDataCollector
from sparse_causal_model_learner_rl.config import Config
from sparse_causal_model_learner_rl.trainable.decoder import Decoder
from sparse_causal_model_learner_rl.trainable.model import Model
from sparse_causal_model_learner_rl.trainable.reconstructor import Reconstructor
from causal_util.helpers import postprocess_info
from matplotlib import pyplot as plt
from causal_util.helpers import dict_to_sacred
from sparse_causal_model_learner_rl.sacred_gin_tune.sacred_wrapper import gin_sacred


@gin.configurable
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

        self.trainables = [self.model, self.decoder, self.reconstructor]

        self.epochs = 0

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
                    obs_y.append(step['observation'])
                    act_x.append(step['action'])

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

        variables = [p for x in self.trainables for p in x.parameters()]
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

        # process epoch information
        epoch_info = postprocess_info(epoch_info)

        # send information downstream
        if self.callback:
            self.callback(self, epoch_info)

        # update config
        self.config.update(epoch_info=epoch_info)
        self.epochs += 1

        return epoch_info

    @property
    def graph(self):
        """Return the current causal model."""
        return None

    def train(self):
        """Train (many epochs)."""
        for _ in tqdm(range(self.config['train_steps'])):
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

    def visualize(self):
        # plotting
        plt.figure(figsize=(16, 5))
        for i, (k_, v) in enumerate(lstdct2dctlst(results).items()):
            plt.subplot(1, len(results[0]) + 1, i + 1)
            plt.xlabel('epoch')
            plt.title(k_)
            plt.axhline(0)
            plt.plot(v)
            plt.yscale('log')

        plt.subplot(1, len(results[0]) + 1, len(results[0]) + 1)
        plt.title("Weights heatmap")
        sns.heatmap(list(model.parameters())[0].detach().numpy())


parser = argparse.ArgumentParser(description="Causal learning experiment")
parser.add_argument('--train', required=False, action='store_true')
parser.add_argument('--config', type=str, default=None, action='append')



if __name__ == '__main__':
    args = parser.parse_args()

    def main_fcn(config, ex, **kwargs):
        """Main function for gin_sacred."""
        def callback(self, epoch_info):
            """Callback for Learner."""
            # pass metrics to sacred
            dict_to_sacred(ex, epoch_info, epoch_info['epochs'])

            # save graph as artifact
            uid = str(uuid.uuid4())
            base_dir = ex.base_dir
            fn = os.path.join(base_dir, f"G_{uid}.pkl")
            pickle.dump(self.graph, open(fn, 'wb'))

            ex.add_artifact(fn, "W")

        learner = Learner(config, callback=callback)
        learner.train()

    if args.train:
        gin_sacred(args.config, main_fcn, db_name='causal_sparse')