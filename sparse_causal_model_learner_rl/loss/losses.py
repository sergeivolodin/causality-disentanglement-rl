import torch
from sparse_causal_model_learner_rl.loss import loss
import gin
import numpy as np


@gin.configurable
def reconstruction_loss(obs, decoder, reconstructor, **kwargs):
    """Ensure that the decoder is not degenerate by fitting a reconstructor."""
    mse = torch.nn.MSELoss()
    return mse(reconstructor(decoder(obs)), obs)

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
def fit_loss(obs_x, obs_y, action_x, decoder, model, **kwargs):
    """Ensure that the model fits the features data."""
    mse = torch.nn.MSELoss()
    # detaching second part like in q-learning makes the loss jitter
    return mse(model(decoder(obs_x), action_x), decoder(obs_y))

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