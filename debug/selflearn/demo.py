# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2
import numpy as np
import torch
from selflearn import Interaction
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

life = Interaction()

life.inference_step()

stats = {'lr': [], 'loss': []}

for _ in tqdm(range(1000)):
    life.selflearn()

    stats['lr'].append(life.lr_out.item())
    stats['loss'].append(life.model.loss.item())

2048 * 1000

plt.plot(stats['lr'])

plt.plot(stats['loss'])

for (x, y) in life.data[0]:
    pred_model = np.argmax(life.model(x.to(life.device)).cpu().detach().numpy(), axis=1)
    out_true = y.cpu().numpy()
    acc = np.mean(pred_model == out_true)
    print(pred_model[:5], out_true[:5], acc)

life.internals

# +
# 0.93, 0.95 2048 bs
# 512 bs explode / 0.9
# -


