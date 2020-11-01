import torch
from sparse_causal_model_learner_rl.loss import loss
import gin
import numpy as np


@gin.configurable
def reconstruction_loss(obs, decoder, reconstructor, **kwargs):
    """Ensure that the decoder is not degenerate by fitting a reconstructor."""
    mse = torch.nn.MSELoss()
    return mse(reconstructor(decoder(obs)), obs)


@gin.configurable
def reconstruction_loss_norm(reconstructor, config, rn_threshold=100, **kwargs):
    """Ensure that the decoder is not degenerate (inverse norm not too high)."""
    regularization_loss = 0
    for param in reconstructor.parameters():
        regularization_loss += torch.sum(torch.square(param))
    if regularization_loss < rn_threshold:
        regularization_loss = torch.from_numpy(np.array(rn_threshold))
    return regularization_loss

@gin.configurable
def reconstruction_loss_inverse_decoder(decoder, rn_threshold, **kwargs):
    regularization_loss = 0
    for param in decoder.parameters():
        regularization_loss += torch.sum(torch.square(torch.pinverse(param)))
    if regularization_loss < rn_threshold:
        regularization_loss = torch.from_numpy(np.array(rn_threshold))
    return regularization_loss

@gin.configurable
def reconstruction_loss_inverse_model(model, rn_threshold, **kwargs):
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(torch.square(torch.pinverse(param)))
    if regularization_loss < rn_threshold:
        regularization_loss = torch.from_numpy(np.array(rn_threshold))
    return regularization_loss


@gin.configurable
def fit_loss(obs_x, obs_y, action_x, decoder, model, **kwargs):
    """Ensure that the model fits the features data."""
    mse = torch.nn.MSELoss()

    return mse(model(decoder(obs_x), action_x), decoder(obs_y))


@gin.configurable
def sparsity_loss(model, **kwargs):
    """Ensure that the model is sparse."""
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(torch.abs(param))
    return regularization_loss