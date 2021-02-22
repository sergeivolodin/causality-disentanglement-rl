import torch
from sparse_causal_model_learner_rl.loss import loss
import gin
import numpy as np
from .helpers import gather_additional_features

@gin.configurable
def MSERelative(pred, target, eps=1e-6):
    """Relative MSE loss."""
    pred = pred.flatten(start_dim=1)
    target = target.flatten(start_dim=1)
    delta = pred - target
    delta = delta
    delta_magnitude = target.std(0).unsqueeze(0)
        
    delta_magnitude = torch.where(delta_magnitude < eps,
                                  torch.ones_like(delta_magnitude),
                                  delta_magnitude)
    
    delta = delta / delta_magnitude    
    delta = delta.pow(2).sum(1).mean()
    return delta

def thr_half(tensor):
    """Get the middle between min/max over batch dimension."""
    m = tensor.min(0, keepdim=True).values
    M = tensor.max(0, keepdim=True).values
    return m, (M - m) / 2.0
    
def delta_01_obs(obs, rec_dec_obs):
    """Compute accuracy between observations and reconstructed observations."""
    obs = torch.flatten(obs, start_dim=1)
    rec_dec_obs = torch.flatten(rec_dec_obs, start_dim=1)
    m1, d1 = thr_half(obs)
    thr1 = m1 + d1
    
    m2, d2 = thr_half(rec_dec_obs)
    thr2 = m2 + d2
    
    delta_01 = 1. * ((obs > thr1) == (rec_dec_obs > thr2))
    delta_01 = torch.where(d1.repeat(obs.shape[0], 1) != 0,
                           delta_01, torch.ones_like(delta_01))
    delta_01_agg = delta_01.mean(1).mean(0)
    return delta_01_agg
    
@gin.configurable
def reconstruction_loss(obs, decoder, reconstructor, relative=False,
                        report_01=True,
                        **kwargs):
    """Ensure that the decoder is not degenerate by fitting a reconstructor."""
    if relative:
        mse = MSERelative
    else:
        mse = lambda x, y: (x - y).flatten(start_dim=1).pow(2).sum(1).mean(0)
        
    rec_dec_obs = reconstructor(decoder(obs))
    metrics = {}
    loss = mse(rec_dec_obs, obs)
    
    if report_01:
        metrics['rec_acc_loss_01_agg'] = 2 - delta_01_obs(obs, rec_dec_obs).item()
        
    return {'loss': loss,
            'metrics': metrics}

def square(t):
    """Torch.square compat."""
    return torch.pow(t, 2.0)

@gin.configurable
def reconstruction_loss_norm(reconstructor, config, rn_threshold=100, **kwargs):
    """Ensure that the decoder is not degenerate (inverse norm not too high)."""
    regularization_loss = 0
    for param in reconstructor.parameters():
        regularization_loss += torch.sum(square(param))
    if regularization_loss < rn_threshold:
        regularization_loss = torch.from_numpy(np.array(rn_threshold))
    return regularization_loss

@gin.configurable
def reconstruction_loss_inverse_decoder(decoder, rn_threshold, **kwargs):
    regularization_loss = 0
    for param in decoder.parameters():
        regularization_loss += torch.sum(square(torch.pinverse(param)))
    if regularization_loss < rn_threshold:
        regularization_loss = torch.from_numpy(np.array(rn_threshold))
    return regularization_loss

@gin.configurable
def reconstruction_loss_inverse_model(model, rn_threshold, **kwargs):
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(square(torch.pinverse(param)))
    if regularization_loss < rn_threshold:
        regularization_loss = torch.from_numpy(np.array(rn_threshold))
    return regularization_loss

@gin.configurable
def reconstruction_loss_value_function_reward_to_go(obs_x, decoder, value_predictor, reward_to_go, value_scaler=1.0, **kwargs):
    mse = torch.nn.MSELoss()
    return mse(value_predictor(decoder(obs_x)), reward_to_go * value_scaler)

@gin.configurable
def manual_switch_gradient(loss_delta_noreduce, model, eps=1e-5):
    """Fill in the gradient of switch probas manually

    Assuming that the batch size is enough to estimate mean loss with
     p on and off.
    """
    mask = model.model.last_mask
    input_dim = model.n_features + model.n_actions
    output_dim = model.n_features + model.n_additional_features

    delta = loss_delta_noreduce

    # if have two dimensions, assuming delta in the form of (batch, n_output_features)
    if len(delta.shape) == 2:
        delta_expanded = delta.view(delta.shape[0], 1, delta.shape[1]).expand(-1, input_dim, -1)
    elif len(delta.shape) == 1:  # assuming shape (batch, )
        delta_expanded = delta.view(delta.shape[0], 1, 1).expand(-1, input_dim, output_dim)
    mask_coeff = (mask - 0.5) * 2

    mask_pos = mask
    mask_neg = 1 - mask
    n_pos = (mask_coeff > 0).sum(dim=0) + eps
    n_neg = (mask_coeff < 0).sum(dim=0) + eps

    mask_pos = mask_pos / n_pos
    mask_neg = mask_neg / n_neg

    mask_atleast = ((n_pos >= 1) * (n_neg >= 1))
    mask_coeff = mask_atleast * (mask_pos - mask_neg)
    p_grad = (delta_expanded * mask_coeff).sum(dim=0)

    if model.model.switch.probas.grad is None:
        model.model.switch.probas.grad = p_grad.clone()
    else:
        model.model.switch.probas.grad += p_grad.clone()
    return 0.0

class MedianStd():
    """Compute median standard deviation of features."""
    def __init__(self, keep_many=100):
        self.last_stds = []
        self.keep_many = keep_many
        
    def forward(self, dataset, save=True, eps=1e-8):
        std = dataset.std(0).detach()
        if not self.last_stds or save:
            self.last_stds.append(std)
            self.last_stds = self.last_stds[-self.keep_many:]
        stds = torch.stack(self.last_stds, dim=0)
        med, _ = torch.median((stds + eps).log(), dim=0)
        med = med.exp()
        return med
    
fit_loss_median_std = MedianStd()

@gin.configurable
def fit_loss(obs_x, obs_y, action_x, decoder, model, additional_feature_keys,
             reconstructor=None,
             report_rec_y=True,
             model_forward_kwargs=None,
             fill_switch_grad=False,
             opt_label=None,
             divide_by_std=True,
             std_eps=1e-6,
             **kwargs):
    """Ensure that the model fits the features data."""

    if model_forward_kwargs is None:
        model_forward_kwargs = {}
    
    f_t1 = decoder(obs_y)
        
    have_additional = False
    if additional_feature_keys:
        have_additional = True
        add_features_y = gather_additional_features(additional_feature_keys=additional_feature_keys,
                                                    **kwargs)
        f_t1 = torch.cat([f_t1, add_features_y], dim=1)

    if divide_by_std:
#         f_t1_std = (fit_loss_median_std.forward(f_t1, save=opt_label is not None).view(1, -1))
#         f_t1_std = torch.max(torch.ones_like(f_t1_std) * std_eps, f_t1_std).detach()
        f_t1_std = f_t1.std(0).unsqueeze(0)
        f_t1_std = torch.where(f_t1_std < std_eps, torch.ones_like(f_t1_std), f_t1_std)
    else:
        f_t1_std = None
        
    # detaching second part like in q-learning makes the loss jitter

    f_t1_pred = model(decoder(obs_x), action_x, all=have_additional, **model_forward_kwargs)

    loss = (f_t1_pred - f_t1).pow(2)
    if f_t1_std is not None:
        loss = loss / f_t1_std.pow(2)

    if fill_switch_grad:
        manual_switch_gradient(loss, model)
        
    loss = loss.mean(1).mean()        

    metrics = {'mean_feature': f_t1.mean(0).detach().cpu().numpy(),
               'std_feature': f_t1.std(0).detach().cpu().numpy(),
               'min_feature': f_t1.min().item(),
               'max_feature': f_t1.max().item(),
               'std_feature_avg': f_t1_std.detach().cpu().numpy() if f_t1_std is not None else 0.0,
               'inv_std_feature_avg': 1/f_t1_std.detach().cpu().numpy() if f_t1_std is not None else 0.0}
    
    if reconstructor is not None and report_rec_y:
        rec_obs_y = reconstructor(f_t1_pred[:, :model.n_features])
        metric_01 = delta_01_obs(obs_y, rec_obs_y)
        loss_rec_y = (rec_obs_y - obs_y).pow(2)
        if f_t1_std is not None:
            loss_rec_y = loss_rec_y / f_t1_std.pow(2)
        loss_rec_y = loss_rec_y.sum(1).mean()
        metrics['rec_fit_y_acc_loss'] = 2 - metric_01.item()
        metrics['rec_fit_y'] = loss_rec_y.item()


    return {'loss': loss,
            'metrics': metrics}

@gin.configurable
def fit_loss_obs_space(obs_x, obs_y, action_x, decoder, model, additional_feature_keys,
             reconstructor,
             model_forward_kwargs=None,
             fill_switch_grad=False,
             opt_label=None,
             add_fcons=True,
             divide_by_std=False,
             loss_local_cache=None,
             std_eps=1e-6,
             **kwargs):
    """Ensure that the model fits the features data."""

    if model_forward_kwargs is None:
        model_forward_kwargs = {}
    
    have_additional = False
    if additional_feature_keys:
        have_additional = True
        add_features_y = gather_additional_features(additional_feature_keys=additional_feature_keys,
                                                    **kwargs)

    
    if 'dec_obs_x' not in loss_local_cache:
        loss_local_cache['dec_obs_x'] = decoder(obs_x)

    f_t1_pred = model(loss_local_cache['dec_obs_x'], action_x, all=have_additional, **model_forward_kwargs)
    f_t1_f = f_t1_pred[:, :model.n_features]
    obs_y_pred = reconstructor(f_t1_f)
    delta_first = obs_y.flatten(start_dim=1)
    delta_second = obs_y_pred.flatten(start_dim=1)

    if additional_feature_keys:
        f_t1_fadd = f_t1_pred[:, model.n_features:]
        delta_first = torch.cat([delta_first, add_features_y], dim=1)
        delta_second = torch.cat([delta_second, f_t1_fadd], dim=1)
    
    delta = delta_first - delta_second

#    if divide_by_std:
#        delta_first_std = delta_first.std(0).unsqueeze(0)
#        delta_first_std = torch.where(delta_first_std < std_eps, torch.ones_like(delta_first_std), delta_first_std)
#    else:
#        delta_first_std = None

    loss = delta.pow(2)
 #   if delta_first_std is not None:
 #       loss = loss / delta_first_std.pow(2)

    loss = loss.sum(1)

    if add_fcons:  # ensure that model(f) ~ f_t1
        f_next_pred = f_t1_f #model(decoder(obs_x).detach(), action_x, all=True, **model_forward_kwargs)
        #f_next_pred = f_next_pred[:, :model.n_features]
        if 'dec_obs_y' not in loss_local_cache:
            loss_local_cache['dec_obs_y'] = decoder(obs_y)
        f_next_true = loss_local_cache['dec_obs_y']#.detach()
        loss_fcons = (f_next_pred - f_next_true).pow(2)
        if divide_by_std:
            delta_fcons_std = f_next_true.std(0).unsqueeze(0)
            delta_fcons_std = torch.where(delta_fcons_std < std_eps, torch.ones_like(delta_fcons_std), delta_fcons_std)
            loss_fcons = loss_fcons / delta_fcons_std.pow(2)
        loss += loss_fcons.sum(1)

    else:
        loss_fcons = None

    if fill_switch_grad:
        manual_switch_gradient(loss, model)
        
    loss = loss.mean(0)        

    metrics = {'mean_feature': f_t1_pred.mean(0).detach().cpu().numpy(),
               'std_feature': f_t1_pred.std(0).detach().cpu().numpy(),
               'min_feature': f_t1_pred.min().item(),
               'max_feature': f_t1_pred.max().item(),
               'loss_fcons': loss_fcons.sum(1).mean(0).item() if loss_fcons is not None else 0.0,
               #'std_obs_avg': delta_first_std.detach().cpu().numpy() if delta_first_std is not None else 0.0,
               #'inv_std_obs_avg': delta_first_std.detach().cpu().numpy() if delta_first_std is not None else 0.0
               }
    metrics['rec_fit_acc_loss_01_agg'] = 2 - delta_01_obs(obs_y, obs_y_pred).item()
    
    return {'loss': loss,
            'metrics': metrics}

def linreg(X, Y):
    """Return weights for linear regression as a differentiable equation."""
    # return torch.pinverse(X.T @ X) @ X.T @ Y
    return torch.pinverse(X) @ Y

def MfMa(obs_x, obs_y, action_x, decoder):
    fx = decoder(obs_x)
    fy = decoder(obs_y)

    fx_ax = torch.cat((fx, action_x), 1)
    # print(fx_ax.shape)
    Mfa = linreg(fx_ax, fy).T

    # sanity check
    assert Mfa.shape[0] == fx.shape[1], (Mfa.shape, fx.shape)
    assert Mfa.shape[1] == fx.shape[1] + action_x.shape[1], (Mfa.shape, fx.shape, action_x.shape)

    Mf = Mfa[:, :fx.shape[1]]
    Ma = Mfa[:, fx.shape[1]:]
    return Mf, Ma

@gin.configurable
def fit_loss_linreg(obs_x, obs_y, action_x, decoder, model, **kwargs):
    """Compute the optimal linear model automatically."""
    Mf, Ma = MfMa(obs_x, obs_y, action_x, decoder)

    # setting the weights directly
    # as if running optimization in 1 step
    model.load_state_dict({'fc_features.weight': Mf, 'fc_action.weight': Ma})

    return torch.abs(torch.tensor(np.array(0.0), requires_grad=True))

@gin.configurable
def sparsity_uniform(tensors, ord, maxval=100.):
    regularization_loss = 0
    # maxval_torch = torch.abs(torch.tensor(torch.from_numpy(np.array(maxval, dtype=np.float32)), requires_grad=False))
    for param in tensors:
        regularization_loss += torch.norm(param.flatten(), p=ord) #torch.min(maxval_torch, torch.norm(param.flatten(), p=ord))
    return regularization_loss / len(tensors)

@gin.configurable
def sparsity_per_tensor(tensors, ord, maxval=100.):
    regularization_loss = 0

    # parameters can have different scale, and we care about number of small elements
    # therefore, dividing by the maximal element

    nparams = 0
    for param in tensors:
        regularization_loss += torch.norm(param.flatten(), p=ord) /\
                               torch.max(torch.abs(param.flatten())) # .detach() doesn't work with linreg optimization (jitter)
        nparams += torch.numel(param)

    # loss=1 means all elements are maximal
    # dividing by the total number of parameters, because care about total number
    # of non-zero arrows
    # and tensors can have different shapes
    return regularization_loss / max(nparams, 1)

@gin.configurable
def sparsity_loss_linreg(obs_x, obs_y, action_x, decoder, fcn=sparsity_uniform, ord=1, **kwargs):
    Mf, Ma = MfMa(obs_x, obs_y, action_x, decoder)
    #return fcn(tensors=[Mf, Ma, torch.pinverse(Mf), torch.pinverse(Ma)], ord=ord)
    return sparsity_uniform([Mf, Ma, torch.inverse(Mf), torch.inverse(Ma)], ord=ord)

@gin.configurable
def nonzero_proba_loss(model, eps=1e-3, do_abs=True, **kwargs):
    """Make probabilities larger than some constant, so that gradients do not disappear completely."""
    params = [x[1] for x in model.sparsify_me()]
    # assuming proba in [0, 1]
    
    params = [param.flatten() for param in params]
    if do_abs:
        params = [param.abs() for param in params]
    
    margin = [torch.nn.ReLU()(eps - param).max() / eps for param in params]
    msp = [p.min().item() for p in params]
    return {'loss': sum(margin) / len(margin),
            'metrics': {'min_switch_proba': min(msp)}}


@gin.configurable
def sparsity_loss(model, device, add_reg=True, ord=1, eps=1e-8, add_inv=True,
                  **kwargs):
    """Ensure that the model is sparse."""
    params = [x[1] for x in model.sparsify_me()]

    def inverse_or_pinverse(M, eps=eps, add_reg=add_reg):
        """Get true inverse if possible, or pseudoinverse."""
        assert len(M.shape) == 2, "Only works for matrices"

        if M.shape[0] != M.shape[1]:
            assert M.shape[1] < M.shape[0], M.shape
            # computing minimal norm of column
            return 1. / (eps + torch.norm(M, p=1, dim=0))

        #if M.shape[0] == M.shape[1]: # can use true inverse
        return torch.inverse(M + torch.eye(M.shape[0], requires_grad=False).to(device) * eps)
        #else:
        #    return torch.pinverse(M, rcond=eps)

    all_params = params
    if add_inv:
        params_inv = [inverse_or_pinverse(p) for p in params]
        all_params += params_inv
    values = {f"sparsity_param_{i}_{tuple(p.shape)}": sparsity_uniform([p], ord=ord) for i, p in enumerate(all_params)}
    return {'loss': sparsity_uniform(all_params, ord=ord),
            'metrics': values}

@gin.configurable
def soft_batchnorm_dec_out(decoder, obs, **kwargs):
    f = decoder(obs)
    lm0 = f.mean(0).pow(2).mean()
    ls1 = (f.std(0) - 1.0).pow(2).mean()

    return lm0 + ls1

@gin.configurable
def soft_batchnorm_std_margin(decoder, obs, margin=1.0, **kwargs):
    f = decoder(obs)
    return torch.nn.ReLU()(margin - f.std(0)).mean()

@gin.configurable
def soft_batchnorm_regul(decoder, **kwargs):
    mse = torch.nn.MSELoss()
    if not hasattr(decoder, 'bn'):
        raise ValueError("Decoder does not have batch norm.")
    bn = decoder.bn
    n = bn.num_features
    params = dict(bn.named_parameters())
    all_zeros = torch.tensor(np.zeros(n), dtype=torch.float32, requires_grad=False)
    regul_loss = mse(torch.log(torch.abs(params['weight'])), all_zeros) + mse(params['bias'], all_zeros)
    return regul_loss
