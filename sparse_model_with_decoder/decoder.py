import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Takes an encoded input and returns the original."""
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x):
        """Take a batch of inputs and decode them."""
        raise NotImplementedError

class FCDecoder(Decoder):
    def __init__(self, input_shape, output_shape, activation=nn.ReLU):
        super(FCDecoder, )