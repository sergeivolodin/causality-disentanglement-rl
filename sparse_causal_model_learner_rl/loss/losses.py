import torch
import logging
from sparse_causal_model_learner_rl.loss import loss
import gin
import numpy as np
from .helpers import gather_additional_features, get_loss_and_metrics


def modify_coeff(kwargs, mult=1.0):
    """Change coefficient of the loss."""
    if 'loss_coeff' in kwargs:
        kwargs_copy = {x: y for x, y in kwargs.items()}
        kwargs_copy['loss_coeff'] = kwargs['loss_coeff'] * mult
        return kwargs_copy
    return kwargs


@gin.configurable
def margin_loss(fcn, margin=1.0, **kwargs):
    l, metrics = get_loss_and_metrics(fcn, **kwargs)
        
    metrics['pre_margin'] = l.item()
    
    l_margin = torch.nn.ReLU()(l - margin)
    
    return {'loss': l_margin, 'metrics': metrics}


@gin.configurable
def linear_combination(losses_dct, **kwargs):
    """Compute a combination of losses."""
    total = 0
    metrics = {}
    for loss_key, loss_dct in losses_dct.items():
        coeff = loss_dct['coeff']
        fcn = loss_dct['fcn']
        c_loss, c_metrics = get_loss_and_metrics(fcn, **modify_coeff(kwargs, coeff))
        metrics[loss_key] = c_metrics
        metrics[loss_key]['coeff'] = coeff
        metrics[loss_key]['value'] = c_loss.item()
        metrics[loss_key]['coeff_value'] = coeff * c_loss.item()
        total += c_loss * coeff
    metrics['total'] = total.item()
    return {'loss': total, 'metrics': metrics}


@gin.configurable
def lagrangian(losses_dict, objective_key, lagrange_multipliers, max_constraint=0.1, mode=None,
               constraint_keys_override=None,
               loss_epoch_cache=None,
               **kwargs):
    assert mode in ['PRIMAL', 'DUAL'], mode
    metrics = {}

    # loss
    # equation: objective_key + lagrange_multipliers * other_losses_linear_combination
    assert lagrange_multipliers.n == 1, lagrange_multipliers.n  # need only 1 component
    lm = lagrange_multipliers()[0]
    
    def get_constraint():
        other_losses_dict = {x: y for x, y in losses_dict.items() if x != objective_key}
        constraint_loss_metrics = linear_combination(other_losses_dict, **modify_coeff(kwargs, lm.item()))
        return constraint_loss_metrics

    def get_constraint_override():
        if constraint_keys_override is None:
            return get_constraint()
        
        other_losses_dict = {x: y for x, y in losses_dict.items() if x in constraint_keys_override}
        constraint_loss_metrics = linear_combination(other_losses_dict, **modify_coeff(kwargs, lm.item()))
        return constraint_loss_metrics
        
    if mode == 'PRIMAL':
        constraint_loss_metrics = cache_get(loss_epoch_cache, key='constraint_cached',
                                            fcn=get_constraint, force=True)
        
        metrics['constraint'] = constraint_loss_metrics['metrics']

        
        c = losses_dict[objective_key]['coeff']
        obj_loss, obj_metrics = get_loss_and_metrics(losses_dict[objective_key]['fcn'], **modify_coeff(kwargs, c))
        obj_metrics['value'] = obj_loss.item()
        metrics['objective_' + objective_key] = obj_metrics
        obj_loss *= c
        
        loss = obj_loss + lm * (constraint_loss_metrics['loss'] - max_constraint)
    elif mode == 'DUAL':
        constraint_loss_metrics = cache_get(loss_epoch_cache, key='constraint_cached_dual',
                                            fcn=get_constraint_override, force=False)
        metrics['constraint'] = constraint_loss_metrics['metrics']
        
        loss = -lm * ((constraint_loss_metrics['loss'] - max_constraint).detach())

    metrics['lagrange_multiplier'] = lm.item()

    return {'loss': loss,
            'metrics': metrics}

@gin.configurable
def lagrangian_granular(
               losses_dict,
               constraints_dict, # map loss key (/ goes into ['losses']) -> constraint: float [None for objective], controlling: bool
               lagrange_multipliers,
               mode=None,
               loss_epoch_cache=None,
               **kwargs):
    assert mode in ['PRIMAL', 'DUAL'], mode
    metrics = {}


    def get_losses():
        result = {}
        for loss_key, loss_dct in losses_dict.items():
            result[loss_key] = {'computed': loss_dct['fcn'](**kwargs),
                                'original': loss_dct}
            for ind_loss_key, ind_loss_val in result[loss_key]['computed'].get('losses', {}).items():
                result[f"{loss_key}/{ind_loss_key}"] = {'computed': {'loss': ind_loss_val, 'metrics': {}},
                                                        'original': loss_dct}

        return result

    losses = cache_get(loss_epoch_cache, '_lagrange_losses', get_losses,
                       force=(mode == 'PRIMAL'))

    total_constraint = 0.0
    total_objective = 0.0

    def maybe_item(z):
        if hasattr(z, 'item'):
            return z.item()
        return z
    def maybe_detach(z):
        if hasattr(z, 'detach'):
            return z.detach()
        return z

    all_losses_lst = list(constraints_dict.keys())
    assert lagrange_multipliers.n == len(all_losses_lst), (lagrange_multipliers.n, all_losses_lst, len(all_losses_lst))

    # filling in the metrics
    for loss_key, loss_dct in losses.items():
        metrics[loss_key] = {'value': maybe_item(loss_dct['computed']['loss']), 'coeff': loss_dct['original']['coeff']}

    # computing the objective and the constraints
    for loss_key, config in constraints_dict.items():
        assert loss_key in losses, (loss_key, losses.keys())
        loss_dct = losses[loss_key]
        current_val_coeff = loss_dct['computed']['loss'] * loss_dct['original']['coeff']
        metrics[loss_key] = loss_dct['computed']['metrics']
        metrics[loss_key]['value'] = maybe_item(loss_dct['computed']['loss'])

        if mode == 'DUAL':
            current_val_coeff = maybe_detach(current_val_coeff)

        if config['constraint'] is None:  # this is the objective
            total_objective += current_val_coeff
        else:

            idx = all_losses_lst.index(loss_key)
            c = config['constraint']
            lm = lagrange_multipliers()[idx]
            if mode == 'PRIMAL':
                lm = lm.detach()
            if not config['controlling']:
                # not using lagrange multiplier
                lm = 1.0

            total_constraint += (current_val_coeff - c) * lm
            metrics['lagrange_multiplier_' + loss_key] = maybe_item(lm)

    # initializing lagrange multipliers
    for loss_key, config in constraints_dict.items():
        loss_dct = losses[loss_key]
        current_val_coeff = loss_dct['computed']['loss'] * loss_dct['original']['coeff']

        if config['constraint'] is not None:  # this is the objective
            idx = all_losses_lst.index(loss_key)
            current_delta = current_val_coeff - config['constraint']
            if lagrange_multipliers.initialized[idx] is False and current_delta > 0:
                new_value = total_objective / current_delta
                new_value = maybe_item(new_value)
                lagrange_multipliers.set_value(idx, new_value)
                logging.warning(f"Objective={total_objective} Loss={current_val_coeff}")
                logging.warning(f"Initializing lagrange multiplier {loss_key} with {new_value}")


    lagrangian = total_objective + total_constraint

    metrics['constraint'] = maybe_item(total_constraint)
    metrics['objective'] = maybe_item(total_objective)
    metrics['lagrangian'] = maybe_item(lagrangian)

    if mode == 'PRIMAL':
        loss = lagrangian
    elif mode == 'DUAL':
        loss = -lagrangian

    return {'loss': loss,
            'metrics': metrics}

def tensor_std(t, eps=1e-8):
    """Compute standard deviation, output 1 if std < eps for stability (disabled features)."""
    s = t.std(0, keepdim=True)
    s = torch.where(s < eps, torch.ones_like(s), s)
    return s

def cache_get(cache, key, fcn, force=False):
    """Get key from cache, or compute one."""
    if cache is None:
        cache = {}
    if force or (key not in cache):
        cache[key] = fcn()
    return cache[key]

@gin.configurable
def MSERelative(pred, target, eps=1e-6):
    """Relative MSE loss."""
    pred = pred.flatten(start_dim=1)
    target = target.flatten(start_dim=1)
    delta = pred - target
    delta = delta

    delta_magnitude = tensor_std(target)

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
            'losses': {'reconstruction': loss},
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
def manual_switch_gradient(loss_delta_noreduce, model, eps=1e-5, loss_coeff=1.0):
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
    
    p_grad *= loss_coeff

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
        f_t1_std = tensor_std(f_t1)

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

def delta_pow2_sum1(true, pred, std_from=None, divide_by_std=False):
    """Compute (relative) MSE error."""
    delta = (true - pred).pow(2)
    if divide_by_std:
        std = tensor_std(true if std_from is None else std_from)
        delta = delta / std.pow(2)
    return delta.sum(1)


@gin.configurable
def fit_loss_obs_space(obs_x, obs_y, action_x, decoder, model, additional_feature_keys,
             reconstructor,
             model_forward_kwargs=None,
             fill_switch_grad=False,
             opt_label=None,
             add_fcons=True,
             obs_relative=False,
             rot_pre=None, rot_post=None,
             divide_by_std=False,
             detach_features=False,
             detach_rotation=False,
             loss_coeff=1.0,
             cross_std=True, # compute std for delta_y from x values
             loss_local_cache=None,
             **kwargs):
    """Ensure that the model fits the features data."""

    #      dec     rot_pre     model           rot_post   rec
    # obs_x -> f_x -> f_x_model -> f_yp_f_model -> f_yp_f -> obs_yp
    # obs_y -> f_y -> f_y_model
    # LOSSES
    # obs_yp ~ obs_y + additional features (f_yp_fadd_model ~ f_y_fadd) [loss]
    # f_yp_f_model ~ f_y_model [loss_fcons_model]
    # f_yp_f ~ f_y

    if rot_pre is None:
        rot_pre = lambda x: x
    if rot_post is None:
        rot_post = lambda x: x

    if model_forward_kwargs is None:
        model_forward_kwargs = {}

    f_x = cache_get(loss_local_cache, 'dec_obs_x',  lambda: decoder(obs_x))

    if detach_features:
        f_x = f_x.detach()

    with torch.set_grad_enabled(not detach_rotation):
        f_x_model = rot_pre(f_x)

    f_yp_model = model(f_x_model, action_x, all=True, **model_forward_kwargs)
    f_yp_f_model = f_yp_model[:, :model.n_features]
    f_yp_f = rot_post(f_yp_f_model)
    obs_yp = reconstructor(f_yp_f)

    loss_rec = delta_pow2_sum1(obs_y.flatten(start_dim=1),
                               obs_yp.flatten(start_dim=1),
                               divide_by_std=obs_relative)
    loss = loss_rec

    if additional_feature_keys:
        f_y_fadd = gather_additional_features(
            additional_feature_keys=additional_feature_keys, **kwargs)
        f_yp_fadd_model = f_yp_model[:, model.n_features:]
        loss_additional = delta_pow2_sum1(f_y_fadd, f_yp_fadd_model)
        loss += loss_additional
    else:
        loss_additional = None


    if add_fcons:
        f_y = cache_get(loss_local_cache, 'dec_obs_y', lambda: decoder(obs_y))

        if detach_features:
            f_y = f_y.detach()

        with torch.set_grad_enabled(not detach_rotation):
            f_y_model = rot_pre(f_y)

        loss_fcons = delta_pow2_sum1(f_y, f_yp_f, divide_by_std=divide_by_std,
                                     std_from=f_x if cross_std else f_y)
        loss += loss_fcons

        loss_fcons_model = delta_pow2_sum1(f_y_model, f_yp_f_model, divide_by_std=divide_by_std,
                                           std_from=f_x_model if cross_std else f_y_model)
        loss += loss_fcons_model
    else:
        loss_fcons = None
        loss_fcons_model = None

    if fill_switch_grad:
        manual_switch_gradient(loss, model, loss_coeff=loss_coeff)

    metrics = {'mean_feature': f_x.mean(0).detach().cpu().numpy(),
               'std_feature': f_x.std(0).detach().cpu().numpy(),
               'min_feature': f_x.min().item(),
               'max_feature': f_x.max().item(),
               'loss_fcons': loss_fcons.mean(0).item() if loss_fcons is not None else 0.0,
               'loss_add': loss_additional.mean(0).item() if loss_additional is not None else 0.0,
               'loss_rec': loss_rec.mean(0).item(),
               'loss_fcons_pre': loss_fcons_model.mean(0).item() if loss_fcons is not None else 0.0,
               'rec_fit_acc_loss_01_agg': 2 - delta_01_obs(obs_y, obs_yp).item(),
               }

    def l_out(l):
        if hasattr(l, 'mean'):
            return l.mean()
        return 0.0
    
    return {'loss': loss.mean(0),
            'losses': {
                'additional': l_out(loss_additional),
                'obs': l_out(loss_rec),
                'feat': l_out(loss_fcons),
                'feat_model': l_out(loss_fcons_model),
            },
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
