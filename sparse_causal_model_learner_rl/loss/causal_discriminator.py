import torch
import gin
from .helpers import gather_additional_features


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
    logits_true_correct = fcn(pair_a=pair_a, pair_b=pair_b)
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
def feature_causal_gan(obs_x, act_x, obs_y, model, decoder,
                       additional_feature_keys,
                       causal_feature_action_model_discriminator,
                       obs_delta_eps=1e-3,
                       loss_type=None,
                       **kwargs):
    """GAN loss where the discriminator classifies correct/incorrect causal pairs."""
    assert loss_type in ['generator', 'discriminator'], f"Wrong loss type {loss_type}"
    batch_dim = len(obs_x)
    device = obs_x.device

    # input features
    f_x = decoder(obs_x)

    # output features
    f_y = decoder(obs_y)
    f_y_add = gather_additional_features(additional_feature_keys=additional_feature_keys, **kwargs)
    f_y_all = torch.cat([f_y, f_y_add], dim=1)

    # permuted output features
    idxes = torch.randperm(batch_dim, device=device)
    f_y_all_permuted = f_y_all[idxes]
    obs_y_permuted = obs_y[idxes]
    act_permuted = act_x[idxes]

    # predicted features
    f_y_pred = model(f_t=f_x, a_t=act_x)

    logits_environment = causal_feature_action_model_discriminator(f_t=f_x, a_t=act_x, f_t1=f_y_all)
    logits_model = causal_feature_action_model_discriminator(f_t=f_x, a_t=act_x, f_t1=f_y_pred)
    logits_permuted = causal_feature_action_model_discriminator(f_t=f_x, a_t=act_x, f_t1=f_y_all_permuted)

    # prediction = 1 -> correct pair
    # prediction = 0 -> incorrect pair
    env_permuted_obs_close = ((obs_y_permuted - obs_y).flatten(start_dim=1).pow(2).sum(1) +
                              (act_permuted - act_x).flatten(start_dimm=1).pow(2).sum(1) <=
                              obs_delta_eps).to(device).detach()
    ans_env = torch.ones((batch_dim,), device=device, dtype=torch.float32)
    ans_env_permuted = env_permuted_obs_close  # act+obs close -> same features -> must output 1, otherwise 0

    if loss_type == 'generator':  # generator wants to have all correct pairs
        ans_model = torch.ones((batch_dim,), device=device, dtype=torch.float32)
    elif loss_type == 'discriminator':  # discriminator must see issues with the generator
        ans_model = torch.zeros((batch_dim,), device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Wrong loss_type: {loss_type}")

    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    loss_env = criterion(logits_environment.view(-1), ans_env)
    loss_env_permuted = criterion(logits_permuted.view(-1), ans_env_permuted)
    loss_model = criterion(logits_model.view(-1), ans_model)

    loss = loss_env + loss_env_permuted + loss_model

    metrics = {
        'loss_env': loss_env.item(),
        'loss_env_permuted': loss_env_permuted.item(),
        'loss_model': loss_model.item(),
        'mean_logits_env': logits_environment.mean(),
        'mean_logits_env_permuted': logits_permuted.mean(),
        'mean_logits_model': logits_model.mean(),
    }

    return {'loss': loss,
            'metrics': metrics}

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