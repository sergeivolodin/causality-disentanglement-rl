import gin
import torch
import logging
from sparse_causal_model_learner_rl.metrics import find_value, find_key



@gin.configurable
def AnnealerThresholdSelector(config, config_object, epoch_info, temp,
                              adjust_every=100,
                              multiplier=10, # allow the loss to be 10 times bigger than the best
                              source_quality_key=None,
                              source_fit_loss_key='no_sparse_fit',
                              gin_variable='ThresholdAnnealer.fit_threshold',
                              **kwargs):
    """Adjust the fit threshold based on a non-sparse model's loss."""
    try:
        non_sparse_fit_loss = find_value(epoch_info, source_fit_loss_key)
        logging.info(f"Threshold detector found non-sparse loss {non_sparse_fit_loss}")
    except AssertionError as e:
        return config

    if 'last_hyper_adjustment' not in temp:
        temp['last_hyper_adjustment'] = 0
    i = epoch_info['epochs']

    temp['suggested_hyper'] = non_sparse_fit_loss * multiplier

    if temp.get('suggested_hyper', None) is not None and (i - temp['last_hyper_adjustment'] >= adjust_every):
        with gin.unlock_config():
            gin.bind_parameter(gin_variable, temp['suggested_hyper'])
        temp['suggested_hyper'] = None
        temp['last_hyper_adjustment'] = i
    return config


@gin.configurable
def ModelResetter(config, epoch_info, temp,
                  learner=None,
                  gin_annealer_cls='ThresholdAnnealer',
                  trainables=None,
                  reset_weights=True,
                  reset_logits=True,
                  reset_optimizers=False,
                  grace_epochs=2000, # give that many epochs to try to recover on its own
                  new_logits=0.0, **kwargs):

    source_metric_key = gin.query_parameter(f"{gin_annealer_cls}.source_metric_key")

    try:
        fit_loss = find_value(epoch_info, source_metric_key)
        # logging.warning("Cannot find loss with sparsity, defaulting to fit loss")
    except AssertionError as e:
        return config

    if 'first_not_good' not in temp:
        temp['first_not_good'] = None

    fit_threshold = gin.query_parameter(f"{gin_annealer_cls}.fit_threshold")
    is_good = fit_loss <= fit_threshold
    i = epoch_info['epochs']

    logging.info(f"Resetter found loss {fit_loss} threshold {fit_threshold}, good {is_good} epoch {i} fng {temp['first_not_good']}")



    if is_good:
        temp['first_not_good'] = None
    elif temp['first_not_good'] is None:
        temp['first_not_good'] = i
    elif i - temp['first_not_good'] >= grace_epochs:
        if reset_weights:
            for key, param in trainables.get('model').named_parameters():
                if 'switch' not in key:
                    logging.info(f'Resetting parameter {key}')
                    if 'bias' in key:
                        torch.nn.init.zeros_(param)
                    else:
                        torch.nn.init.xavier_uniform_(param)

        if reset_logits:
            for p in trainables.get('model').switch__params:
                logging.info(f"Resetting switch parameter with shape {p.data.shape}")
                p_orig = p.data.detach().clone()
                p.data[1, p_orig[1] < -new_logits] = -new_logits
                p.data[0, p_orig[1] < -new_logits] = new_logits

        if reset_optimizers:
            learner.create_optimizers()
        temp['first_not_good'] = None

@gin.configurable
def ThresholdAnnealer(config, epoch_info, temp,
                      fit_threshold=1e-2,
                      min_hyper=1e-5,
                      max_hyper=100,
                      adjust_every=100,
                      reset_on_fail=False,
                      source_metric_key='with_sparse_fit',
                      factor=0.5, # if cool/warm not specified, use this one for both
                      factor_cool=None, # when increasing the coefficient (regularization -> cooling)
                      factor_heat=None, # when decreasing the coefficient (no reg -> warming)
                      emergency_heating=False,
                      **kwargs):
    """Increase sparsity if fit loss is low, decrease otherwise."""

    try:
        fit_loss = find_value(epoch_info, source_metric_key)
        # logging.warning("Cannot find loss with sparsity, defaulting to fit loss")
        logging.info(f"Annealer found loss {fit_loss} {source_metric_key}")
    except AssertionError as e:
        #logging.warning(f"Annealer source metric not found: {source_metric_key}, {e}")
        return config
        # fit_loss = find_value(epoch_info, '/fit/value')

    if factor_cool is None:
        factor_cool = factor
    if factor_heat is None:
        factor_heat = factor

    need_heating = False

    if 'last_hyper_adjustment' not in temp:
        temp['last_hyper_adjustment'] = 0
    i = epoch_info['epochs']

    if fit_loss > fit_threshold: # FREE ENERGY (loss) IS HIGH -> NEED WARMING (decrease regul coeff)
        if reset_on_fail:
            temp['suggested_hyper'] = min_hyper
        else:
            if config['losses']['sparsity']['coeff'] > min_hyper:
                temp['suggested_hyper'] = config['losses']['sparsity']['coeff'] * factor_heat
                need_heating = True
                temp['suggested_hyper'] = max(min_hyper, temp['suggested_hyper'])

    else: # FREE ENRGY (loss) is low -> CAN DO COOLING (increase regul coeff)
        if config['losses']['sparsity']['coeff'] < max_hyper:
            temp['suggested_hyper'] = config['losses']['sparsity']['coeff'] / factor_cool
            temp['suggested_hyper'] = min(max_hyper, temp['suggested_hyper'])

    epochs_enough = (i - temp['last_hyper_adjustment'] >= adjust_every)
    if emergency_heating and need_heating:
        epochs_enough = True

    if temp.get('suggested_hyper', None) is not None and epochs_enough:
        config['losses']['sparsity']['coeff'] = temp['suggested_hyper']
        temp['suggested_hyper'] = None
        temp['last_hyper_adjustment'] = i
    return config


@gin.configurable
def threshold_annealer_threshold(**kwargs):
    return gin.query_parameter('ThresholdAnnealer.fit_threshold')