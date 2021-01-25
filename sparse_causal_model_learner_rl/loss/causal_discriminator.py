import torch
import gin


@gin.configurable
def decoder_discriminator_loss(obs, decoder, decoder_discriminator, **kwargs):
    """True value is not 0, sometimes will get the same input..."""
    features = decoder(obs)
    batch_dim = obs.shape[0]

    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    # original inputs order
    idxes_orig = torch.range(batch_dim)

    # random permutation for incorrect inputs
    idxes = torch.randperm(batch_dim)
    obs_shuffled = obs[idxes]

    # correct pairs for contrastive loss
    logits_true_correct = decoder_discriminator(o_t=obs, f_t=features)
    target_correct = torch.ones([batch_dim, ], dtype=torch.float32)

    # incorrect pairs, contrastive loss
    logits_true_incorrect = decoder_discriminator(o_t=obs_shuffled, f_t=features)
    target_incorrect = idxes == idxes_orig

    # two parts of the loss
    loss_correct = criterion(logits_true_correct, target_correct)
    loss_incorrect = criterion(logits_true_incorrect, target_incorrect)

    return loss_correct + loss_incorrect