from tqdm.auto import tqdm
import numpy as np
import tensorflow as tf
from causal_util.collect_data import EnvDataCollector
from matplotlib import pyplot as plt
import os
import sys
from sparse_causal_model_learner_rl.learners.rl_learner import CausalModelLearnerRL as Learner
import gin
from sparse_causal_model_learner_rl.sacred_gin_tune.sacred_wrapper import load_config_files
from sparse_causal_model_learner_rl.config.config import Config
from sparse_causal_model_learner_rl.trainable.decoder import IdentityDecoder
import pickle
from torch.utils.data import TensorDataset, DataLoader
import torch
import cv2
from keychest.keychestenv import obss_to_rgb as obss_to_rgb_orig

prefix = os.path.join(os.path.dirname(__file__), '..')
load_config_files([prefix + '/../keychest/config/5x5.gin', prefix + '/configs/common.gin'])

def get_xy_conv(steps=1000, orig_shape=True):
    """Get the dataset."""
    gin.bind_parameter('Config.feature_shape', None)
    gin.bind_parameter('KeyChestEnvironment.flatten_observation', not orig_shape)
    gin.bind_parameter('Config.env_steps', steps)
    gin.bind_parameter('Config.decoder', None)
    gin.bind_parameter('Config.model', None)
    gin.bind_parameter('Config.reconstructor', None)
    gin.bind_parameter('Config.value_predictor', None)
    gin.bind_parameter('Config.disable_cuda', True)
    learner = Learner(Config())
    learner.collect_steps(do_tqdm=True)
    obs_x = learner._context.get('obs_x').cpu().numpy()
    obs_y = learner._context.get('obs_y').cpu().numpy()
    act_x = learner._context.get('action_x').cpu().numpy()
    
    if orig_shape:
        return obs_x, act_x, obs_y
    else:
        X = np.concatenate((obs_x, act_x), axis=1)
        y = obs_y
        return X, y

learner = Learner(Config())
engine = learner.env.engine

env = learner.env
h, w, c = env.engine._observation.shape

def obss_to_rgb(obss, engine=engine):
    return obss_to_rgb_orig(obss, engine)

@gin.configurable
def rgb_resize(obss_rgb, size_x=32, size_y=32, active=False):
    if not active:
        return obss_rgb
    res = [cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_NEAREST) for img in obss_rgb]
    return np.array(res)

@gin.configurable
def rgb_pad(obss_rgb, size_x=16, size_y=8):
    b, x, y, c = obss_rgb.shape
    x1 = max(x, size_x)
    y1 = max(y, size_y)
    out = np.zeros((b, x1, y1, c), dtype=obss_rgb.dtype)
    out[:, :x, :y, :] = obss_rgb
    return out

def plot_data(x, gx=3, gy=8, figsize=(15, 10)):
    fig = plt.figure(figsize=figsize)

    idx_start = np.random.choice(len(x) - gx * gy - 1)

    for i in range(gx * gy):
        plt.subplot(gx, gy, i + 1)
        plt.imshow(x[idx_start + i])
    return fig

@gin.configurable
def get_dataloader(steps=100000, batch_size=512):
    Xo_train, Xa_train, yo_train = get_xy_conv(steps=steps, orig_shape=True)

    #  see https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader
    Xo_train_torch = torch.Tensor(np.rollaxis(rgb_pad(rgb_resize(obss_to_rgb(Xo_train))), 3, 1))
    Xa_train_torch = torch.Tensor(Xa_train)
    yo_train_torch = torch.Tensor(np.rollaxis(rgb_pad(rgb_resize(obss_to_rgb(yo_train))), 3, 1))

    my_dataset = TensorDataset(Xo_train_torch, Xa_train_torch, yo_train_torch) # create your datset
    # my_dataset = TensorDataset(Xo_train_torch, Xa_train_torch, Xo_train_torch) # PURE RECONSTRUCTION
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)#False) # create your dataloader
    return my_dataloader
