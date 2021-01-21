import gin
import torch

@gin.configurable
def pow_loss(X, X_true, power=2, **kwargs):
    """Difference between items to a power, avg over dataset, sum over items."""
    assert X.shape == X_true.shape, (X.shape, X_true.shape)
    abs_diff = torch.abs(X - X_true)
    loss = torch.pow(abs_diff, power)

    # mean over batches
    loss = torch.mean(loss, dim=0)

    # sum over the rest
    loss = torch.sum(loss)

    return loss


@gin.configurable
def ae_loss(autoencoder, X_chw, loss_fcn, **kwargs):
    """Vanilla autoencoder loss."""
    reconstructed = autoencoder(X_chw)
    value = loss_fcn(X_chw, reconstructed)
    return value