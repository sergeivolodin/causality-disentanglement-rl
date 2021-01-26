import gin
import torch
from torch import nn


@gin.configurable
class Discriminator(nn.Module):
    """Discriminate between inputs. Outputs LOGITS!"""

    def __init__(self, input_shapes_dict, **kwargs):
        super(Discriminator, self).__init__()
        self.input_shapes_dict = input_shapes_dict
        self.inputs = sorted(self.input_shapes_dict.keys())
        self.output_shape = (1,)

    def model(self, **kwargs):
        return NotImplementedError

    def forward(self, **kwargs):
        for key in self.input_shapes_dict.keys():
            assert key in kwargs, f"Required input not found: {key} {kwargs}"
            assert kwargs[key].shape[1:] == self.input_shapes_dict[key], f"Input shape is wrong" \
                                                                         f" for {key}: {kwargs[key].shape} {self.input_shapes_dict[key]}"
        batch_dims = [kwargs[key].shape[0] for key in self.input_shapes_dict.keys()]
        assert all([batch_dims[0] == batch_dim for batch_dim in
                    batch_dims]), f"Batch dimensions must be equal: {batch_dims}"

        return self.model(**kwargs)


@gin.configurable
class ModelDiscriminator(Discriminator):
    """Compute embeddings for inputs, and output aggregated value."""

    def __init__(self, input_embedder_cls, aggregator_cls,
                 input_embedding_dims, **kwargs):
        super(ModelDiscriminator, self).__init__(**kwargs)
        self.models = {}

        self.input_embedder_cls = input_embedder_cls
        self.aggregator_cls = aggregator_cls
        self.input_embedding_dims = input_embedding_dims
        self.total_embedding_dim = sum(self.input_embedding_dims.values())

        for inp_name, inp_shape in self.input_shapes_dict.items():
            assert inp_name in self.input_embedder_cls
            assert inp_name in self.input_embedding_dims

            emb_cls = self.input_embedder_cls[inp_name]
            emb_dim = self.input_embedding_dims[inp_name]

            m = emb_cls(input_shape=inp_shape, output_shape=(emb_dim,))
            setattr(self, f'model_{inp_name}', m)
            self.models[inp_name] = m

        self.agg = self.aggregator_cls(input_shape=(self.total_embedding_dim,),
                                       output_shape=(1,))
        setattr(self, 'aggregator', self.agg)

    def model(self, **kwargs):
        embeddings = [self.models[inp_name](kwargs[inp_name]) for inp_name in self.inputs]
        embeddings = torch.cat(embeddings, dim=1)
        # print("EMB", embeddings, kwargs[self.inputs[0]].shape, self.inputs[0])
        assert embeddings.shape[1] == self.total_embedding_dim
        assert embeddings.shape[0] == kwargs[self.inputs[0]].shape[0],\
            (embeddings.shape, kwargs[self.inputs[0]].shape)
        out = self.agg(embeddings)
        # out = nn.Sigmoid()(out) # OUTPUTS LOGITS
        return out


@gin.configurable
class CausalFeatureModelDiscriminator(ModelDiscriminator):
    """Discriminate between correct next features and wrong next features."""

    def __init__(self, feature_shape, feature_embedding_dim=10, **kwargs):
        self.feature_shape = feature_shape
        self.feature_embedding_dim = feature_embedding_dim
        super(CausalFeatureModelDiscriminator, self) \
            .__init__(input_embedding_dims={'f_t': self.feature_embedding_dim,
                                            'f_t1': self.feature_embedding_dim},
                      input_shapes_dict={'f_t': self.feature_shape,
                                         'f_t1': self.feature_shape},
                      **kwargs)


@gin.configurable
class DecoderDiscriminator(ModelDiscriminator):
    """Discriminate between correct features and wrong features for observations."""

    def __init__(self, observation_shape, feature_shape,
                 observation_embedding_dim, feature_embedding_dim,
                 **kwargs):
        self.feature_shape = feature_shape
        self.observation_shape = observation_shape
        self.observation_embedding_dim = observation_embedding_dim
        self.feature_embedding_dim = feature_embedding_dim

        super(DecoderDiscriminator, self).__init__(
            input_shapes_dict={
                'o_t': self.observation_shape,
                'f_t': self.feature_shape
            },
            input_embedding_dims={
                'o_t': self.observation_embedding_dim,
                'f_t': self.feature_embedding_dim
            },
            **kwargs
        )

@gin.configurable
class DifferenceAggregator(nn.Module):
    """Aggregate embeddings with an l2 difference."""
    def __init__(self, input_shape, output_shape):
        super(DifferenceAggregator, self).__init__()
        assert len(input_shape) == 1, input_shape
        assert len(output_shape) == 1, output_shape
        assert output_shape[0] == 1, output_shape
        assert input_shape[0] % 2 == 0, input_shape
        self.input_dim_half = input_shape[0] // 2
        self.loss = nn.MSELoss()
        self.to_logits = nn.Linear(1, 1)

    def forward(self, x):
        p1 = x[:, :self.input_dim_half]
        p2 = x[:, self.input_dim_half:]
        return self.to_logits(self.loss(p1, p2))