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
from causal_util.helpers import lstdct2dctlst

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
    def __init__(self, intermediate_size=128, hidden_size=20, action_size=4,
                 input_channels=3, hid_x=8, hid_y=4, kernel_sizes=[3, 2, 3, 3],
                 strides=[1, 2, 1, 1], paddings=[1, 0, 1, 1],
                 channels=[3, 32, 32, 32], last_conv=True,
                 add_gan=False):
        super(ObsNet, self).__init__()
        # Encoder
        hidden_size2 = action_size + hidden_size
        
        self.add_gan = add_gan
        
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
            setattr(self, 'enc_conv%02d' % i, nn.Conv2d(self.channels[i - 1], self.channels[i], **layer_kwargs[i - 1]))
        
        if add_gan:
            # creating convolutional layers
            for i in range(1, len(layer_kwargs) + 1):
                setattr(self, 'gan_conv%02d' % i, nn.Conv2d(self.channels[i - 1], self.channels[i], **layer_kwargs[i - 1]))
            self.gan_fc1 = nn.Linear(self.hid_x * self.hid_y * self.channels[-1], hidden_size)
            self.gan_fc2 = nn.Linear(hidden_size, 1)
        
        def list_params(prefix):
            f = lambda x: x.startswith(prefix)
            return [param for layer in filter(f, dir(self)) for param in getattr(self, layer).parameters()]
        
            
        # space before computing mean and logstd
        self.enc_fc1 = nn.Linear(self.hid_x * self.hid_y * self.channels[-1], intermediate_size)
        
        # Latent space
        self.enc_fc21 = nn.Linear(intermediate_size, hidden_size)
        self.enc_fc22 = nn.Linear(intermediate_size, hidden_size)

        # Decoder
        self.dec_fc3 = nn.Linear(hidden_size2, intermediate_size)
        self.dec_fc4 = nn.Linear(intermediate_size, self.hid_x * self.hid_y * self.channels[-1])
        
        # creating deconvolutional layers
        for i in range(1, len(layer_kwargs))[::-1]:
            setattr(self, 'dec_deconv%02d' % i, nn.ConvTranspose2d(self.channels[i + 1], self.channels[i], **layer_kwargs[i]))
        
        # last convolutional layer to apply after deconvolutional layers
        last_cls = nn.Conv2d if last_conv else nn.ConvTranspose2d
        self.dec_conv_last = last_cls(self.channels[1], self.channels[0], **layer_kwargs[0])

        self.params_dis = list_params('gan_')
        self.params_enc = list_params('enc_')
        self.params_dec = list_params('dec_')

        
        # activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def apply_conv(self, out):
        results = [out]
        for conv_fcn in sorted(filter(lambda x: x.startswith('enc_conv'), dir(self))):
            out = self.relu(getattr(self, conv_fcn)(out))
            results.append(out)
        return results
    
    def apply_gan(self, out):
        out = self.apply_conv_gan(out)[-1]
        out = out.reshape(out.size(0), -1)
        out = self.relu(self.gan_fc1(out))
        out = self.sigmoid(self.gan_fc2(out))
        return out
    
    def apply_conv_gan(self, out):
        results = [out]
        for conv_fcn in sorted(filter(lambda x: x.startswith('gan_conv'), dir(self))):
            out = self.relu(getattr(self, conv_fcn)(out))
            results.append(out)
        return results
    
    def apply_deconv(self, out):
        results = [out]
        for deconv_fcn in sorted(filter(lambda x: x.startswith('dec_deconv'), dir(self)), reverse=True):
            out = self.relu(getattr(self, deconv_fcn)(out))
            results.append(out)
        results.append(self.dec_conv_last(out))
        return results
        
    def encode(self, x):
        out = self.apply_conv(x)[-1]
        assert out.shape[-2:] == (self.hid_x, self.hid_y), (out.shape, self.hid_x, self.hid_y)
        out = out.reshape(out.size(0), -1)
        h1 = self.relu(self.enc_fc1(out))
        return h1, self.enc_fc21(h1), self.enc_fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.dec_fc3(z))
        out = self.relu(self.dec_fc4(h3))
        out = out.view(out.size(0), self.channels[-1], self.hid_x, self.hid_y)
        out = self.apply_deconv(out)[-1]
        out = self.sigmoid(out)
        return out

    def forward(self, x, a):
        h1, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        za = torch.cat([z,a], 1)
        return self.decode(za), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        assert recon_x.shape == x.shape, (recon_x.shape, x.shape)
        BCE = F.binary_cross_entropy(recon_x.reshape(-1, img_x * img_y * self.channels[0]),
                                     x.reshape(-1, img_x * img_y * self.channels[0]), size_average=False) # does not work without size_average=False???
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KLD
        mae = torch.nn.L1Loss()(recon_x, x)
        mse = torch.nn.MSELoss()(recon_x, x)
        
        loss_dict = {'bce': BCE, 'kld': KLD, 'mae': mae, 'mse': mse, 'vae': BCE + KLD}

        
        if self.add_gan:
            discriminate_true = self.apply_gan(x)
            discriminate_gen  = self.apply_gan(recon_x)
            GAN_true = -torch.mean(torch.log(discriminate_true))
            GAN_gen = -torch.mean(torch.log(1 - discriminate_gen))
            loss_dict['gan_true'] = GAN_true
            loss_dict['gan_gen'] = GAN_gen
            GAN = GAN_true + GAN_gen
            loss_dict['gan'] = GAN
            loss += GAN
        
        loss_dict['total'] = loss
        loss_dict = {x: y.item() for x, y in loss_dict.items()}
            
        return loss, loss_dict

@gin.configurable
class ObsModel(object):
    def __init__(self, train_loader, eval_loader, optimizer_cls=optim.Adam,
                optimizer_dis_cls=optim.Adam, optimizer_dec_cls=optim.Adam,
                 optimizer_enc_cls=optim.Adam):
        self.model = ObsNet().cuda()
        self.optimizer = optimizer_cls(params=self.model.parameters())
        
        if self.model.add_gan:
            print("creating optimizers")
            self.optimizer_dis = optimizer_dis_cls(params=self.model.params_dis)
            self.optimizer_enc = optimizer_enc_cls(params=self.model.params_enc)
            self.optimizer_dec = optimizer_dec_cls(params=self.model.params_dec)
            
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        
    def train(self):
        return self.iterate(self.train_loader, 'train')

    def eval(self):
        return self.iterate(self.eval_loader, 'eval')
    
    def iterate(self, dataset, mode):
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()
        else:
            raise NotImplementedError
            
        def get_loss_withdict(states, actions, next_states):
            recon_batch, mu, logvar = self.model(states, actions)
            loss, loss_dict = self.model.loss_function(recon_batch, next_states, mu, logvar)
            return loss, loss_dict
            
        losses = []
        for i, data in enumerate(dataset):

            if len(data) == 3:
                (states, actions, next_states) = data
                
                states = Variable(states).cuda()
                next_states = Variable(next_states).cuda()
                actions = Variable(actions).cuda()
            
            else:
                states = data[0].cuda()
                actions = torch.zeros((states.shape[0], 4)).cuda()
                next_states = states

            if mode == 'eval':
                _, loss_dict = get_loss_withdict(states, actions, next_states)
                
            if mode == 'train':
                if not self.model.add_gan:
                    self.optimizer.zero_grad()

                    loss, loss_dict = get_loss_withdict(states, actions, next_states)

                    loss.backward()
                    self.optimizer.step()
                else:
                    for opt_label in ['dec', 'enc', 'dis']:
                        opt = getattr(self, f"optimizer_{opt_label}")
                        opt.zero_grad()
                        
                        loss, loss_dict = get_loss_withdict(states, actions, next_states)
                        
                        loss.backward()
                        opt.step()
            
            losses.append(loss_dict)
            
        losses = lstdct2dctlst(losses)
        losses = {f"{mode}_{x}": np.mean(y) for x, y in losses.items()}
        return losses