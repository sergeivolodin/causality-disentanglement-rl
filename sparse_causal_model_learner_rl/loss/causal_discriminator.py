import torch
import gin

def contrastive_loss_permute(pair_a, pair_b, fcn, invert_labels=False):
    """Contrastive loss by permuting pair_a."""
    assert pair_a.shape[0] == pair_b.shape[0], (pair_a.shape, pair_b.shape)
    batch_dim = pair_a.shape[0]

    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    # original inputs order
    idxes_orig = torch.arange(start=0, end=batch_dim).to(pair_a.device)

    # random permutation for incorrect inputs
    idxes = torch.randperm(batch_dim).to(pair_a.device)
    pair_a_shuffled = pair_a[idxes]

    # correct pairs for contrastive loss
    logits_true_correct = fcn(pair_a=pair_a, pair_b=psair_b)
    target_correct = torch.ones([batch_dim, ], dtype=torch.float32).to(pair_a.device)

    # incorrect pairs, contrastive loss
    logits_true_incorrect = fcn(pair_a=pair_a_shuffled, pair_b=pair_b)
    target_incorrect = (idxes == idxes_orig).to(torch.float32).to(pair_a.device)

    if invert_labels:
        target_correct = 1 - target_correct
        target_incorrect = 1 - target_incorrect

    mean_logits_correct = logits_true_correct.mean()
    mean_logits_incorrect = logits_true_incorrect.mean()

    # two parts of the loss
    loss_correct = criterion(logits_true_correct.view(-1), target_correct)
    loss_incorrect = criterion(logits_true_incorrect.view(-1), target_incorrect)

    return {'loss': loss_correct + loss_incorrect,
            'metrics': {'disc_correct': loss_correct.item(),
                        'disc_incorrect': loss_incorrect.item(),
                        'mean_logits_correct': mean_logits_correct.item(),
                        'mean_logits_incorrect': mean_logits_incorrect.item(),
                        'mean_incorrect_collision': target_incorrect.mean().item()}}


@gin.configurable
def decoder_discriminator_loss(obs, decoder, decoder_discriminator, **kwargs):
    """True value is not 0, sometimes will get the same input..."""

    def fcn(pair_a, pair_b):
        # pair_a == obs
        return decoder_discriminator(o_t=pair_a, f_t=pair_b)

    return contrastive_loss_permute(obs, decoder(obs), fcn)


@gin.configurable
def siamese_feature_discriminator(obs, decoder, causal_feature_model_discriminator, **kwargs):
    def fcn(pair_a, pair_b):
        return causal_feature_model_discriminator(f_t=pair_a, f_t1=pair_b)

    return contrastive_loss_permute(decoder(obs), decoder(obs), fcn, invert_labels=False)

@gin.configurable
def siamese_feature_discriminator_l2(obs, decoder, obs_delta_eps=1e-3,
                                     margin=0.5,
#                                      margin_same_not_farther=5.0,
                                     max_dist=500,
                                     **kwargs):
    def loss(y_true, y_pred):
        """L2 norm for the distance, no flat."""
        delta = y_true - y_pred
        delta = delta.pow(2)
        delta = delta.flatten(start_dim=1)
        delta = delta.sum(1)
        return delta

    # original inputs order
    batch_dim = obs.shape[0]

    # random permutation for incorrect inputs
    idxes = torch.randperm(batch_dim).to(obs.device)
    obs_shuffled = obs[idxes]

    idxes_orig = torch.arange(start=0, end=batch_dim).to(obs.device)
    target_close = ((obs - obs_shuffled).flatten(start_dim=1).pow(2).sum(1) <=\
                        obs_delta_eps).to(obs.device).detach()

    # distance_shuffle = loss(obs, obs_shuffled)
    distance_f = loss(decoder(obs), decoder(obs_shuffled))

    # print(torch.nn.ReLU()(margin - distance_f), torch.where)

#     torch.where(target_close,
# #                                 torch.nn.ReLU()(distance_f - margin_same_not_farther),
#                                 torch.nn.ReLU()(margin_different_not_closer - distance_f[~])
#                                ).mean(),
    
    loss = torch.nn.ReLU()(margin - distance_f[~target_close]).mean()
    loss_max_dist = torch.nn.ReLU()(distance_f - max_dist).mean()
    
    return {'loss': loss.mean() / margin + loss_max_dist.mean() / max_dist,
            'metrics': {'distance_close': distance_f[target_close].mean().item(),
                        'distance_far': distance_f[~target_close].mean().item()}
            }