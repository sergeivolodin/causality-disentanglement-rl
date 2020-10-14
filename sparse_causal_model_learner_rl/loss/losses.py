import torch
from sparse_causal_model_learner_rl.loss import loss


@loss
def reconstruction_loss(observations, decoder, reconstructor, **kwargs):
    """Ensure that the decoder is not degenerate by fitting a reconstructor."""
    mse = torch.nn.MSELoss()
    return mse(reconstructor(decoder(observations)), observations)


@loss
def reconstruction_loss_norm(reconstructor, config, **kwargs):
    """Ensure that the decoder is not degenerate (inverse norm not too high)."""
    regularization_loss = 0
    for param in reconstructor.parameters():
        regularization_loss += torch.sum(torch.square(param))
    if regularization_loss < config['rn_threshold']:
        regularization_loss = torch.from_numpy(np.array(config['rn_threshold']))
    return regularization_loss


@loss
def fit_loss(obs_x, obs_y, action_x, decoder, model, **kwargs):
    """Ensure that the model fits the features data."""
    mse = torch.nn.MSELoss()

    return mse(model(decoder(obs_x), action_x), decoder(obs_y))


@loss
def sparsity_loss(model, **kwargs):
    """Ensure that the model is sparse."""
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(torch.abs(param))
    return regularization_loss