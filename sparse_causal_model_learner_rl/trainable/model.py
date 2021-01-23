import gin
from torch import nn
import numpy as np
import torch


@gin.configurable
class Model(nn.Module):
    def __init__(self, feature_shape, action_shape, **kwargs):
        super(Model, self).__init__()
        self.feature_shape = feature_shape
        if len(self.feature_shape) == 0:
            self.feature_shape = (1,)
        self.action_shape = action_shape
        if len(self.action_shape) == 0:
            self.action_shape = (1,)

    def forward(self, f_t, a_t):
        return NotImplementedError

    def sparsify_me(self):
        """List of (name, tensor) to sparsify in some way."""
        return []


class ManyNetworkModel(nn.Module):
    """Instantiate many networks, each modelling one feature."""
    def __init__(self, skip_connection=True, model_cls=None, sparse_do_max=True, **kwargs):
        super(ManyNetworkModel, self).__init__(**kwargs)
        assert len(self.feature_shape) == 1, f"Features must be scalar: {self.feature_shape}"
        assert len(self.action_shape) == 1, f"Actions must be scalar: {self.action_shape}"

        self.n_features = self.feature_shape[0]
        self.n_actions = self.action_shape[0]
        self.model_cls = model_cls

        self.models = [model_cls(input_shape=(self.n_features + self.n_actions,),
                                 output_shape=(1,)) for _ in range(self.n_features)]
        self.skip_connection = skip_connection
        self.sparse_do_max = sparse_do_max

    @property
    def Mf(self):
        """Return features model."""

        weights = [x[1][:self.n_features].detach().cpu().numpy()
                   for x in self.sparsify_me(sparse_do_max=True)]
        weights = np.array(weights)
        assert weights.shape == (self.n_features, self.n_features)
        return weights

    @property
    def Ma(self):
        """Return action model."""
        weights = [x[1][self.n_features:].detach().cpu().numpy()
                   for x in self.sparsify_me(sparse_do_max=True)]
        weights = np.array(weights)
        assert weights.shape == (self.n_features, self.n_actions)
        return weights

    def sparsify_me(self, sparse_do_max=None):
        """List of sparsifiable (name, tensor), max-ed over output dimension."""
        if sparse_do_max is None:
            sparse_do_max = self.sparse_do_max
        for name, w in self.sparsify_tensors():
            if sparse_do_max:
                wmax = torch.max(torch.abs(w), dim=0)
                assert wmax.shape[0] == self.n_features + self.n_actions
                yield name, wmax
            else:
                yield name, w


    def sparsify_tensors(self):
        """List of named tensors to sparsify."""
        for m in self.models:
            name, w = list(m.named_parameters())[0]
            assert name.find('weight') >= 0
            assert w.shape[1] == self.n_features + self.n_actions
            yield (name, w)


    def forward(self, f_t, a_t):
        assert f_t.shape[1] == self.n_features, f"Wrong f_t shape {f_t.shape}"
        assert a_t.shape[1] == self.n_actions, f"Wrong a_t shape {a_t.shape}"
        assert f_t.shape[0] == a_t.shape[0], f"Wrong batches {f_t.shape} {a_t.shape}"

        # features and actions together
        fa_t = torch.cat((f_t, a_t), dim=1)

        # all models on data
        f_t1 = [m(fa_t) for m in self.models]

        # predictions as a tensor
        f_t1 = torch.cat(f_t1, dim=1)

        # sanity check for output
        assert f_t1.shape[1] == self.n_features, f"Must return n_features {f_t1.shape}"
        assert f_t1.shape[0] == f_t.shape[0], f"Wrong out batches {f_t.shape} {f_t1.shape}"

        if self.skip_connection:
            f_t1 += f_t

        return f_t1


@gin.configurable
class LinearModel(Model):
    def __init__(self, use_bias=True, init_identity=False, **kwargs):
        super(LinearModel, self).__init__(**kwargs)
        self.use_bias = use_bias
        assert len(self.feature_shape) <= 1, f"Features must be scalar: {self.feature_shape}"
        assert len(self.action_shape) <= 1, f"Actions must be scalar: {self.action_shape}"
        self.fc_features = nn.Linear(self.feature_shape[0], self.feature_shape[0], bias=self.use_bias)
        self.fc_action = nn.Linear(self.action_shape[0], self.feature_shape[0], bias=False)
        if init_identity:
            self.fc_features.weight.data.copy_(torch.eye(self.feature_shape[0]))
            self.fc_action.weight.data.copy_(torch.eye(self.feature_shape[0], self.action_shape[0]))

    def forward(self, f_t, a_t):
        f_next_f = self.fc_features(f_t)
        f_next_a = self.fc_action(a_t)
        return f_next_f + f_next_a

    def sparsify_me(self):
        """List of sparsifiable (name, tensor), max-ed over output dimension."""
        return self.named_parameters()

    @property
    def Mf(self):
        """Return features model."""
        return list(self.fc_features.parameters())[0].detach().cpu().numpy()

    @property
    def Ma(self):
        """Return action model."""
        return list(self.fc_action.parameters())[0].detach().cpu().numpy()

@gin.configurable
class ModelModel(Model):
    """Model of the environment which uses a torch model to make predictions."""
    def __init__(self, model_cls, skip_connection=False, init_zeros=False, **kwargs):
        super(ModelModel, self).__init__(**kwargs)
        assert len(self.feature_shape) == 1, f"Features must be scalar: {self.feature_shape}"
        assert len(self.action_shape) == 1, f"Actions must be scalar: {self.action_shape}"
        self.model = model_cls(input_shape=(self.feature_shape[0] + self.action_shape[0], ),
                               output_shape=self.feature_shape)
        self.skip_connection = skip_connection
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.constant_(m.weight, 0.0)
        #        torch.nn.init.constant_(m.bias, 0.0)
        if init_zeros:
            self.model.apply(init_weights)


    def forward(self, f_t, a_t):
        fa_t = torch.cat((f_t, a_t), dim=1)
        f_t1 = self.model(fa_t)
        if self.skip_connection:
            f_t1 += f_t
        return f_t1
