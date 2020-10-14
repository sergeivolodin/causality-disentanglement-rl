import torch

def reconstruction_loss(obss, decoder, reconstructor, **kwargs):
    """Ensure that the decoder is not degenerate by fitting a reconstructor."""
    mse = torch.nn.MSELoss()
    return mse(reconstructor(decoder(obss)), obss)


def reconstruction_loss_norm(reconstructor):
    """Ensure that the decoder is not degenerate (inverse norm not too high)."""
    regularization_loss = 0
    for param in reconstructor.parameters():
        regularization_loss += torch.sum(torch.square(param))
    return regularization_loss


def fit_loss(obss, decoder, model):
    """Ensure that the model fits the features data."""
    mse = torch.nn.MSELoss()

    return mse(model(decoder(obss[:-1])), decoder(obss[1:]))


def sparsity_loss(model):
    """Ensure that the model is sparse."""
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(torch.abs(param))
    return regularization_loss


# %%

def losses(obss_torch, decoder, reconstructor, model, rn_threshold=100):
    """Compute all losses ob observations."""
    res = {}
    res['r'] = reconstruction_loss(obss_torch, decoder, reconstructor)
    res['f'] = fit_loss(obss_torch, decoder, model)
    res['s'] = sparsity_loss(model)
    res['rn'] = reconstruction_loss_norm(reconstructor)
    if res['rn'] < rn_threshold:
        res['rn'] = torch.from_numpy(np.array(rn_threshold))
    return res


def total_loss(losses_, hypers):
    """Compute total loss."""
    loss = 0.0
    for key in hypers.keys():
        loss += hypers[key] * losses_[key]
    return loss