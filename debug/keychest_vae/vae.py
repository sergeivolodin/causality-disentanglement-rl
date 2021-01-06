#!/usr/bin/env python
# taken from https://github.com/bharatprakash/vae-dynamics-model/blob/master/env_model/obs_model.py
import gin
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import sys
import os
import numpy as np

img_x = 16
img_y = 8

def to_img(x):
    x = x.view(x.size(0), 3, img_x, img_y)
    return x

def to_onehot(size, value):
    my_onehot = np.zeros((size))
    my_onehot[value] = 1.0
    return my_onehot

gin.enter_interactive_mode()

@gin.configurable
class ObsNet(nn.Module):
    """Create a network encoding observations."""
    def __init__(self, intermediate_size=128, hidden_size=20, hidden_size2=24,
                 input_channels=3, hid_x=8, hid_y=4, kernel_sizes=[3, 2, 3, 3],
                 strides=[1, 2, 1, 1], paddings=[1, 0, 1, 1],
                 channels=[3, 32, 32, 32]):
        super(ObsNet, self).__init__()
        # Encoder
        
        # hidden image size in the middle of the network
        self.hid_x = hid_x
        self.hid_y = hid_y

        assert len(kernel_sizes) == len(strides) == len(paddings) == len(channels)
        
        # parameters for convolutional/deconvolutional networks
        layer_kwargs_fields = ['kernel_size', 'stride', 'padding']
        l = locals()
        layer_kwargs = [
            {field: l[field + 's'][i] for field in layer_kwargs_fields} for i in range(len(kernel_sizes))
        ]
        
        # number of channels for convolutional layers
        self.channels = [input_channels] + channels
        
        assert len(layer_kwargs) == len(self.channels) - 1
        
        # creating convolutional layers
        for i in range(1, len(layer_kwargs) + 1):
            setattr(self, 'conv%02d' % i, nn.Conv2d(self.channels[i - 1], self.channels[i], **layer_kwargs[i - 1]))
        
        # space before computing mean and logstd
        self.fc1 = nn.Linear(self.hid_x * self.hid_y * self.channels[-1], intermediate_size)
        
        # Latent space
        self.fc21 = nn.Linear(intermediate_size, hidden_size)
        self.fc22 = nn.Linear(intermediate_size, hidden_size)

        # Decoder
        self.fc3 = nn.Linear(hidden_size2, intermediate_size)
        self.fc4 = nn.Linear(intermediate_size, self.hid_x * self.hid_y * self.channels[-1])
        
        # creating deconvolutional layers
        for i in range(1, len(layer_kwargs))[::-1]:
            setattr(self, 'deconv%02d' % i, nn.ConvTranspose2d(self.channels[i + 1], self.channels[i], **layer_kwargs[i]))
        
        # last convolutional layer to apply after deconvolutional layers
        self.reconstruct_conv_last = nn.Conv2d(self.channels[1], self.channels[0], **layer_kwargs[0])

        # activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        out = x
        for conv_fcn in sorted(filter(lambda x: x.startswith('conv'), dir(self))):
            out = self.relu(getattr(self, conv_fcn)(out))
        assert out.shape[-2:] == (self.hid_x, self.hid_y), (out.shape, self.hid_x, self.hid_y)
        out = out.view(out.size(0), -1)
        h1 = self.relu(self.fc1(out))
        return h1, self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        out = self.relu(self.fc4(h3))
        out = out.view(out.size(0), self.channels[-1], self.hid_x, self.hid_y)
        for deconv_fcn in sorted(filter(lambda x: x.startswith('deconv'), dir(self)), reverse=True):
            out = self.relu(getattr(self, deconv_fcn)(out))
        out = self.sigmoid(self.reconstruct_conv_last(out))
        return out

    def forward(self, x, a):
        h1, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        za = torch.cat([z,a], 1)
        return self.decode(za), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        assert recon_x.shape == x.shape
        BCE = F.binary_cross_entropy(recon_x.view(-1, img_x * img_y * self.channels[0]),
                                     x.view(-1, img_x * img_y * self.channels[0]), size_average=False) # does not work without size_average=False???
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


class ObsModel(object):
    def __init__(self, train_loader, eval_loader, optimizer_cls=optim.Adam):
        self.model = ObsNet().cuda()
        self.optimizer = optimizer_cls(self.model.parameters())
        self.train_loader = train_loader
        self.eval_loader = eval_loader

    def train(self):
        self.model.train()
        train_loss = 0
        train_mae = 0
        for i, data in enumerate(self.train_loader):

            if len(data) == 3:
                (states, actions, next_states) = data
                
                states = Variable(states).cuda()
                next_states = Variable(next_states).cuda()
                actions = Variable(actions).cuda()
            
            else:
                states = data[0].cuda()
                actions = torch.zeros((states.shape[0], 4)).cuda()
                next_states = states

            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(states, actions)
            loss = self.model.loss_function(recon_batch, next_states, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            train_mae += torch.nn.L1Loss()(recon_batch, next_states).item()

        N = len(self.train_loader) * self.train_loader.batch_size
        print('====> Average loss: {} MAE: {}'.format(
              train_loss/N, train_mae / len(self.train_loader)))
        return {'train_loss': train_loss / N, 'train_mae': train_mae / len(self.train_loader)}

    def eval(self):
        self.model.eval()
        eval_loss = 0
        eval_mae = 0
        for i, data in enumerate(self.eval_loader):

            if len(data) == 3:
                (states, actions, next_states) = data
                
                states = Variable(states).cuda()
                next_states = Variable(next_states).cuda()
                actions = Variable(actions).cuda()
            
            else:
                states = data[0].cuda()
                actions = torch.zeros((states.shape[0], 4)).cuda()
                next_states = states

            recon_batch, mu, logvar = self.model(states, actions)
            loss = self.model.loss_function(recon_batch, next_states, mu, logvar)
            eval_loss += loss.item()
            eval_mae += torch.nn.L1Loss()(recon_batch, next_states).item()

        N = len(self.eval_loader) * self.eval_loader.batch_size
        print('====> Average loss: {} MAE: {}'.format(
              eval_loss/N, eval_mae / len(self.eval_loader)))
        return {'eval_loss': eval_loss / N, 'eval_mae': eval_mae / len(self.eval_loader)}
