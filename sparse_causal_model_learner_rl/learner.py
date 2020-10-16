import gin

from sparse_causal_model_learner_rl.trainable.decoder import Decoder
from sparse_causal_model_learner_rl.trainable.model import Model
from sparse_causal_model_learner_rl.trainable.reconstructor import Reconstructor
from sparse_causal_model_learner_rl.config import Config
from causal_util import load_env
from causal_util.collect_data import EnvDataCollector
import argparse


@gin.configurable
class Learner(object):
    def __init__(self, config):
        assert isinstance(config, Config), f"Please supply a valid config: {config}"
        self.config = config

        # creating environment
        self.env = self.create_env()
        self.collector = EnvDataCollector(self.env)

        self.feature_shape = self.config.get('feature_shape')
        self.action_shape = self.env.action_space.shape
        self.observation_shape = self.env.observation_space.shape

        # self.action_shape = self.config.get('action_shape')
        # self.observation_shape = self.config.get('observation_shape')

        self.model_cls = config.get('model')
        assert issubclass(self.model_cls, Model)
        self.model = self.model_cls(feature_shape=self.feature_shape,
                                    action_shape=self.action_shape)

        self.decoder_cls = config.get('decoder')
        assert issubclass(self.decoder_cls, Decoder)
        self.decoder = self.decoder_cls(feature_shape=self.feature_shape,
                                        observation_shape=self.observation_shape)

        self.reconstructor_cls = config.get('reconstructor')
        assert issubclass(self.reconstructor_cls, Reconstructor)
        self.reconstructor = self.reconstructor_cls(feature_shape=self.feature_shape,
                                                    observation_shape=self.observation_shape)



    def create_env(self):
        """Create an environment according to config."""
        if 'env_config_file' in self.config:
            gin.parse_config_file(self.config['env_config_file'])
        return load_env()

    def collect_steps(self):
        # collecting data
        n_steps = self.config.get('env_steps')
        while self.collector.steps < n_steps:
            done = False
            self.collector.reset()
            while not done:
                _, _, done, _ = self.collector.step(self.collector.action_space.sample())

    def _epoch(self):
        # obtain data from environment
        self.collect_steps()

        # train using losses
        # pass metrics to sacred
        # save graph as artifact


        # update config
        self.config.update()

    @property
    def graph(self):
        pass

    def train(self):
        pass

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
parser.add_argument('--config', type=str, default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    gin.parse_config_file(args.config)
    learner = Learner(Config())

    if args.train():
        learner.train()