import gin

from sparse_causal_model_learner_rl.trainable.decoder import Decoder
from sparse_causal_model_learner_rl.trainable.model import Model
from sparse_causal_model_learner_rl.trainable.reconstructor import Reconstructor
from causal_util import load_env


@gin.configurable
class Learner(object):
    def __init__(self, config):
        self.config = config

        self.feature_shape = self.config.get('feature_shape')
        self.action_shape = self.config.get('action_shape')
        self.observation_shape = self.config.get('observation_shape')

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

    def _epoch(self):
        pass

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

if __name__ == '__main__':
    pass