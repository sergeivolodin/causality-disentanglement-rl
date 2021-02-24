import gin
import torch
import logging
from sparse_causal_model_learner_rl.metrics import find_value, find_key



@gin.configurable
def AnnealerThresholdSelector(config, config_object, epoch_info, temp,
                              adjust_every=100,
                              multiplier=10, # allow the loss to be 10 times bigger than the best
                              source_quality_key=None,
                              non_sparse_threshold_disable=None,
                              additive=True,
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

    if additive:
        temp['suggested_hyper'] = non_sparse_fit_loss + multiplier
    else:
        temp['suggested_hyper'] = non_sparse_fit_loss * multiplier

    # disable annealing in case if target performance in terms of non-sparse loss is not reached
    if non_sparse_threshold_disable is not None and non_sparse_fit_loss >= non_sparse_threshold_disable:
        temp['suggested_hyper'] = 0.0

    if temp.get('suggested_hyper', None) is not None and (i - temp['last_hyper_adjustment'] >= adjust_every):
        with gin.unlock_config():
            gin.bind_parameter(gin_variable, temp['suggested_hyper'])
        temp['suggested_hyper'] = None
        temp['last_hyper_adjustment'] = i
    return config

@gin.configurable
def turn_on_features(m, ctx, logits_on=1.5, gap_threshold=1.1, loss_fcn=None):
    """Turn on features giving better loss."""
    with torch.no_grad():
        for fout in range(m.n_features + m.n_additional_features):
            if fout >= m.n_features:
                fout_add = fout - m.n_features
                logits = getattr(m, m.additional_models[fout_add]).switch.logits
            else:
                logits = getattr(m, m.models[fout]).switch.logits

            for fin in range(m.n_features):
                orig_logits0, orig_logits1 = logits[0, fin].item(), logits[1, fin].item()

                # trying 0...
                logits[0, fin], logits[1, fin] = 5, -5
                loss_0 = loss_fcn(**ctx)
                if isinstance(loss_0, dict):
                    loss_0 = loss_0['loss']
                loss_0 = loss_0.item()


                # trying 1...
                logits[0, fin], logits[1, fin] = -5, 5
                loss_1 = loss_fcn(**ctx)
                if isinstance(loss_1, dict):
                    loss_1 = loss_1['loss']
                loss_1 = loss_1.item()

                logits[0, fin], logits[1, fin] = orig_logits0, orig_logits1

                loss_ratio = loss_0 / loss_1

                if loss_ratio > gap_threshold:
                    logging.info(f'Turn on feature {fout} <- {fin}')
                    logits[0, fin], logits[1, fin] = -logits_on, logits_on


@gin.configurable
def ModelResetter(config, epoch_info, temp,
                  learner=None,
                  gin_annealer_cls='ThresholdAnnealer',
                  trainables=None,
                  reset_weights=True,
                  reset_logits=True,
                  reset_optimizers=False,
                  grace_epochs=2000, # give that many epochs to try to recover on its own
                  last_context=None,
                  reset_turn_on=False,
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
            
        if reset_turn_on:
            turn_on_features(m=learner.model, ctx=last_context)
            
        temp['first_not_good'] = None

@gin.configurable
def ThresholdAnnealer(config, epoch_info, temp,
                      fit_threshold=1e-2,
                      min_hyper=1e-5,
                      learner=None,
                      max_hyper=100,
                      freeze_time=100,
                      freeze_threshold_probas=0.8,
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

    if temp.get('last_freeze_start', -1) >= 0:
        if i - temp.get('last_freeze_start') >= freeze_time:
            logging.warning(f"Freezing finished at {i}!")
            del temp['last_freeze_start']
        else:
            if freeze_threshold_probas is not None:
                p = learner.model.model.switch.probas
                p.data[p.data > freeze_threshold_probas] = freeze_threshold_probas
            return config

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
        if temp['suggested_hyper'] < config['losses']['sparsity']['coeff']:
            direction = 'heat'
        elif temp['suggested_hyper'] > config['losses']['sparsity']['coeff']:
            direction = 'cool'
        else:
            direction = 'same'

        # if were cooling down but now have to warm...
        # freezing the model for some time
        if 'last_direction' in temp and temp['last_direction'] in ['cool', 'same'] and direction == 'heat':
            temp['last_freeze_start'] = i
            logging.warning(f"Starting model freeze at {i}")

        temp['last_direction'] = direction

        config['losses']['sparsity']['coeff'] = temp['suggested_hyper']
        temp['suggested_hyper'] = None
        temp['last_hyper_adjustment'] = i
    return config


@gin.configurable
def threshold_annealer_threshold(**kwargs):
    return gin.query_parameter('ThresholdAnnealer.fit_threshold')
