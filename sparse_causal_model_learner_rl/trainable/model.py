import gin
from torch import nn
import numpy as np
import torch
# import threading


@gin.configurable
class Model(nn.Module):
    def __init__(self, feature_shape, action_shape, additional_feature_shape, **kwargs):
        super(Model, self).__init__()

        self.feature_shape = feature_shape
        self.additional_feature_shape = additional_feature_shape
        if len(self.feature_shape) == 0:
            self.feature_shape = (1,)
        self.action_shape = action_shape
        if len(self.action_shape) == 0:
            self.action_shape = (1,)
        if len(self.additional_feature_shape) == 0:
            self.additional_feature_shape = (1,)

    def forward(self, f_t, a_t, additional=False, all=False):
        return NotImplementedError

    def sparsify_me(self):
        """List of (name, tensor) to sparsify in some way."""
        return []


@gin.configurable
class ManyNetworkModel(Model):
    """Instantiate many networks, each modelling one feature."""
    def __init__(self, skip_connection=True, model_cls=None, sparse_do_max=True,
                 sparse_do_max_mfma=True, **kwargs):
        super(ManyNetworkModel, self).__init__(**kwargs)
        assert len(self.feature_shape) == 1, f"Features must be scalar: {self.feature_shape}"
        assert len(self.action_shape) == 1, f"Actions must be scalar: {self.action_shape}"
        assert len(self.additional_feature_shape) == 1, f"Additional features must be scalar: {self.additional_feature_shape}"

        self.n_features = self.feature_shape[0]
        self.n_actions = self.action_shape[0]
        self.n_additional_features = self.additional_feature_shape[0]
        self.model_cls = model_cls

        self.models = []
        self.additional_models = []

        self.skip_connection = skip_connection
        self.sparse_do_max = sparse_do_max
        self.sparse_do_max_mfma = sparse_do_max_mfma

        for f in range(self.n_features):
            m = model_cls(input_shape=(self.n_features + self.n_actions,),
                          output_shape=(1,))
            m_name = 'model_fout_%04d' % f
            setattr(self, m_name, m)
            self.models.append(m_name)

        for f in range(self.n_additional_features):
            m = model_cls(input_shape=(self.n_features + self.n_actions,),
                          output_shape=(1,))
            m_name = 'model_fadd_%04d' % f
            setattr(self, m_name, m)
            self.additional_models.append(m_name)

        self.all_models = self.models + self.additional_models

        for m in self.all_models:
            getattr(self, m).share_memory()

    @property
    def model__params(self):
        """List of model (not switch) parameters."""
        for m in self.all_models:
            m = getattr(self, m)
            if hasattr(m, 'model') and hasattr(m, 'switch'):
                for p in m.model.parameters():
                    yield p
            else:
                for p in m.parameters():
                    yield p
    @property
    def switch__params(self):
        """List of switch parameters."""
        for m in self.all_models:
            m = getattr(self, m)
            if hasattr(m, 'model') and hasattr(m, 'switch'):
                for p in m.switch.parameters():
                    yield p
            else:
                for p in m.parameters():
                    yield p

    @property
    def Mf(self):
        """Return features model."""

        weights = [x[1][:self.n_features].detach().cpu().numpy()
                   for x in self.sparsify_me(sparse_do_max=self.sparse_do_max_mfma)]
        weights = np.array(weights)
        assert weights.shape == (self.n_features + self.n_additional_features, self.n_features)
        return weights

    @property
    def Ma(self):
        """Return action model."""
        weights = [x[1][self.n_features:].detach().cpu().numpy()
                   for x in self.sparsify_me(sparse_do_max=self.sparse_do_max_mfma)]
        weights = np.array(weights)
        assert weights.shape == (self.n_features + self.n_additional_features, self.n_actions)
        return weights

    def sparsify_me(self, sparse_do_max=None):
        """List of sparsifiable (name, tensor), max-ed over output dimension."""
        if sparse_do_max is None:
            sparse_do_max = self.sparse_do_max

        for name, w in self.sparsify_tensors():
            if sparse_do_max:
                wmax, _ = torch.max(torch.abs(w), dim=0)
                assert wmax.shape[0] == self.n_features + self.n_actions
                yield name, wmax
            else:
                yield name, w


    def sparsify_tensors(self):
        """List of named tensors to sparsify."""
        for mname in self.all_models:
            m = getattr(self, mname)
            if hasattr(m, 'sparsify_me'):
                # Gumbel-Softmax model
                for name, w in m.sparsify_me():
                    name = mname + '.' + name
                    assert w.shape[0] == self.n_features + self.n_actions
                    yield (name, w)

            else:
                name, w = list(m.named_parameters())[0]
                name = mname + '.' + name
                assert name.find('weight') >= 0
                assert w.shape[1] == self.n_features + self.n_actions
                yield (name, w)


    def forward(self, f_t, a_t, additional=False, all=False, **kwargs):

        n_f_out = self.n_additional_features if additional else self.n_features
        if all:
            n_f_out = self.n_features + self.n_additional_features

        assert f_t.shape[1] == self.n_features, f"Wrong f_t shape {f_t.shape}"
        assert a_t.shape[1] == self.n_actions, f"Wrong a_t shape {a_t.shape}"
        assert f_t.shape[0] == a_t.shape[0], f"Wrong batches {f_t.shape} {a_t.shape}"

        # features and actions together
        fa_t = torch.cat((f_t, a_t), dim=1)

        use_models = self.additional_models if additional else self.models
        if all:
            use_models = self.models + self.additional_models

        # all models on data
        # def set_model_future(fut, model, input_data):
        #     fut.set_result(model(input_data))
        # f_t1 = [getattr(self, m)(fa_t) for m in use_models]

        # futs = [torch.futures.Future() for _ in use_models]
        # threads = [threading.Thread(target=set_model_future, args=(fut, getattr(self, m), fa_t))
        #            for (fut, m) in zip(futs, use_models)]
        # f_t1 = [fut.wait() for fut in futs]
        # [t.join() for t in threads]
        fa_t.share_memory_()
        f_t1 = torch.nn.parallel.parallel_apply([getattr(self, m) for m in use_models],
                                                [fa_t] * len(use_models),
                                                kwargs_tup=[kwargs] * len(use_models))

        # predictions as a tensor
        f_t1 = torch.cat(f_t1, dim=1)

        # sanity check for output
        assert f_t1.shape[1] == n_f_out, f"Must return {n_f_out} features add={additional}: {f_t1.shape}"
        assert f_t1.shape[0] == f_t.shape[0], f"Wrong out batches {f_t.shape} {f_t1.shape}"

        if self.skip_connection and not additional and not all:
            f_t1 += f_t

        return f_t1


@gin.configurable
class ManyNetworkCombinedModel(Model):
    """Instantiate many networks, each modelling one feature."""
    def __init__(self, model_cls=None, sparse_do_max=True,
                 sparse_do_max_mfma=True, **kwargs):
        super(ManyNetworkCombinedModel, self).__init__(**kwargs)
        assert len(self.feature_shape) == 1, f"Features must be scalar: {self.feature_shape}"
        assert len(self.action_shape) == 1, f"Actions must be scalar: {self.action_shape}"
        assert len(self.additional_feature_shape) == 1, f"Additional features must be scalar: {self.additional_feature_shape}"

        self.n_features = self.feature_shape[0]
        self.n_actions = self.action_shape[0]
        self.n_additional_features = self.additional_feature_shape[0]
        self.n_total_features = self.n_features + self.n_additional_features
        self.model_cls = model_cls

        self.sparse_do_max = sparse_do_max
        self.sparse_do_max_mfma = sparse_do_max_mfma

        self.model = model_cls(input_shape=(self.n_features + self.n_actions,),
                               output_shape=(1,),
                               n_models=self.n_total_features)
    @property
    def model__params(self):
        """List of model (not switch) parameters."""
        m = self.model
        if hasattr(m, 'model') and hasattr(m, 'switch'):
            for p in m.model.parameters():
                yield p
        else:
            for p in m.parameters():
                yield p
    @property
    def switch__params(self):
        """List of switch parameters."""
        m = self.model
        if hasattr(m, 'model') and hasattr(m, 'switch'):
            for p in m.switch.parameters():
                yield p
        else:
            for p in m.parameters():
                yield p

    @property
    def Mf(self):
        """Return features model."""

        weights = [x[1][:self.n_features].detach().cpu().numpy()
                   for x in self.sparsify_me(sparse_do_max=self.sparse_do_max_mfma)]
        assert len(weights) == 1
        # shape: [input_features, output_features]
        weights = weights[0].T

        assert weights.shape == (self.n_features + self.n_additional_features, self.n_features)
        return weights

    @property
    def Ma(self):
        """Return action model."""
        weights = [x[1][self.n_features:].detach().cpu().numpy()
                   for x in self.sparsify_me(sparse_do_max=self.sparse_do_max_mfma)]
        assert len(weights) == 1
        # shape: [input_features, output_features]
        weights = weights[0].T
        assert weights.shape == (self.n_features + self.n_additional_features, self.n_actions)
        return weights

    def sparsify_me(self, sparse_do_max=None):
        """List of sparsifiable (name, tensor), max-ed over output dimension."""
        if sparse_do_max is None:
            sparse_do_max = self.sparse_do_max

        for name, w in self.sparsify_tensors():
            if sparse_do_max:
                wmax, _ = torch.max(torch.abs(w), dim=0)
                assert wmax.shape[0] == self.n_features + self.n_actions, (wmax.shape, self.n_features, self.n_actions)
                yield name, wmax
            else:
                yield name, w


    def sparsify_tensors(self):
        """List of named tensors to sparsify."""
        m = self.model
        mname = 'model'
        if hasattr(m, 'sparsify_me'):
            # Gumbel-Softmax model
            for name, w in m.sparsify_me():
                name = mname + '.' + name
                assert w.shape[0] == self.n_features + self.n_actions
                yield (name, w)

        else:
            # print(self.model)
            raise NotImplementedError

    def forward(self, f_t, a_t, additional=False, all=False, **kwargs):
        assert all is True
        n_f_out = self.n_total_features

        assert f_t.shape[1] == self.n_features, f"Wrong f_t shape {f_t.shape}"
        assert a_t.shape[1] == self.n_actions, f"Wrong a_t shape {a_t.shape}"
        assert f_t.shape[0] == a_t.shape[0], f"Wrong batches {f_t.shape} {a_t.shape}"

        # features and actions together
        fa_t = torch.cat((f_t, a_t), dim=1)

        f_t1 = self.model(fa_t, **kwargs)

        # sanity check for output
        assert f_t1.shape[1] == n_f_out, f"Must return {n_f_out} features add={additional}: {f_t1.shape}"
        assert f_t1.shape[0] == f_t.shape[0], f"Wrong out batches {f_t.shape} {f_t1.shape}"

        return f_t1

@gin.configurable
class LinearModel(Model):
    def __init__(self, use_bias=True, init_identity=False, **kwargs):
        super(LinearModel, self).__init__(**kwargs)
        self.use_bias = use_bias
        assert len(self.feature_shape) <= 1, f"Features must be scalar: {self.feature_shape}"
        assert len(self.action_shape) <= 1, f"Actions must be scalar: {self.action_shape}"
        assert len(self.additional_feature_shape) <= 1, f"Additional features must be scalar: {self.additional_feature_shape}"

        self.fc_features = nn.Linear(self.feature_shape[0], self.feature_shape[0], bias=self.use_bias)
        self.fc_action = nn.Linear(self.action_shape[0], self.feature_shape[0], bias=False)

        self.additional_models = False
        if self.additional_feature_shape[0]:
            self.fc_features_additional = nn.Linear(self.feature_shape[0], self.additional_feature_shape[0], bias=self.use_bias)
            self.fc_action_additional = nn.Linear(self.action_shape[0], self.additional_feature_shape[0], bias=False)
            self.additional_models = True

        if init_identity:
            self.fc_features.weight.data.copy_(torch.eye(self.feature_shape[0]))
            self.fc_action.weight.data.copy_(torch.eye(self.feature_shape[0], self.action_shape[0]))


    def forward(self, f_t, a_t, additional=False, all=False):

        if all:
            f_next_f_add = self.fc_features_additional(f_t)
            f_next_a_add = self.fc_action_additional(a_t)
            f_next_f_main = self.fc_features(f_t)
            f_next_a_main = self.fc_action(a_t)
            f_next_f = torch.cat([f_next_f_main, f_next_f_add], dim=1)
            f_next_a = torch.cat([f_next_a_main, f_next_a_add], dim=1)
        else:
            if additional:
                f_next_f = self.fc_features_additional(f_t)
                f_next_a = self.fc_action_additional(a_t)
            else:
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
        assert len(self.additional_feature_shape) == 1, f"Additional features must be scalar: {self.additional_feature_shape}"

        self.model = model_cls(input_shape=(self.feature_shape[0] + self.action_shape[0], ),
                               output_shape=self.feature_shape)

        self.additional_models = False
        if self.additional_feature_shape[0]:
            self.additional_models = True
            self.additional_model = model_cls(input_shape=(self.feature_shape[0] + self.action_shape[0],),
                                               output_shape=self.additional_feature_shape)

        self.skip_connection = skip_connection
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.constant_(m.weight, 0.0)
        #        torch.nn.init.constant_(m.bias, 0.0)
        if init_zeros:
            self.model.apply(init_weights)


    def forward(self, f_t, a_t, additional=False, all=False, **kwargs):
        fa_t = torch.cat((f_t, a_t), dim=1)

        if all:
            f_t1_main = self.model(fa_t, **kwargs)
            f_t1_add = self.additional_model(fa_t, **kwargs)
            f_t1 = torch.cat([f_t1_main, f_t1_add], dim=1)
        else:
            if additional:
                f_t1 = self.additional_model(fa_t, **kwargs)
            else:
                f_t1 = self.model(fa_t, **kwargs)
                if self.skip_connection:
                    f_t1 += f_t
        return f_t1
