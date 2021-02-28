import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm.auto import tqdm
import pytest
import torch
from torch import nn
import gin
from sparse_causal_model_learner_rl.trainable.combined import FCCombinedModel

use_cuda = False

def get_model():
    m = torch.nn.Sequential(
        torch.nn.Linear(in_features=24, out_features=60,),
        torch.nn.Tanh(),
        torch.nn.Linear(in_features=60, out_features=60),
        torch.nn.Tanh(),
        torch.nn.Linear(in_features=60, out_features=60),
        torch.nn.Tanh(),
        torch.nn.Linear(in_features=60, out_features=1),
    )
    return m

class AllModels(nn.Module):
    def __init__(self, n_models):
        super(AllModels, self).__init__()

        self.models = []
        self.model_list = []
        for m in range(n_models):
            m_name = 'm%02d' % m
            model = get_model()
            setattr(self, m_name, model)
            self.models.append(m_name)
            self.model_list.append(model)

    def forward(self, data, do_parallel=False):
        results = [getattr(self, m)(data) for m in self.models]
        return torch.cat(results, dim=1)

@pytest.mark.parametrize("n_models", [10, 20, 30])
def test_combined_inp_outp(n_models):

    data = torch.randn(1000, 24)
    M = AllModels(n_models)
    C = FCCombinedModel(hidden_sizes=[60, 60, 60], input_shape=(24,), n_models=n_models, output_shape=(1,),
                        activation_cls=torch.nn.Tanh)
    
    def print_with_shape(dct):
        dct = dict(dct)
        dct_shape = {x: y.shape for x, y in dct.items()}
        print(dct_shape)
    
    print_with_shape(M.named_parameters())
    print_with_shape(C.named_parameters())

    for n, p in M.named_parameters():
        model_n = int(n[1:3])
        layer_id = int(n.split('.')[1]) // 2
        is_bias = 'bias' in n
        print(n, model_n, is_bias, layer_id)
        target_param = C.fc[layer_id]
        if is_bias:
            target_param.bias.data[:, model_n] = p
        else:
            target_param.weight.data[:, :, model_n] = p

    outC = C(data.view(-1, 24, 1).expand(-1, -1, n_models))
    outM = M(data)
    
    outC = outC.detach().cpu().numpy()
    outM = outM.detach().cpu().numpy()
    
    print(outC.mean(), outM.mean())
    
    delta = np.abs(outC - outM)
    delta = np.mean(delta)
    
    print(delta)

    assert delta < 1e-7