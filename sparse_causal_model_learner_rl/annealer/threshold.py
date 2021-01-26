import gin
import logging


def find_key(dct, key_substr):
    """Get a key from a dictionary containing a substring."""
    keys = dct.keys()
    keys_match = [k for k in keys if key_substr in k]
    if len(keys_match) > 1:
        keys_match = keys_match[:1]
    assert len(keys_match) == 1, f"Unknown substring {key_substr} in {keys}, appears {len(keys_match)} times"
    return keys_match[0]


def find_value(dct, key_substr):
    """Get a value from a dict given a substring in a key."""
    return dct[find_key(dct, key_substr)]


@gin.configurable
def AnnealerThresholdSelector(config, config_object, epoch_info, temp,
                              adjust_every=100,
                              multiplier=10, # allow the loss to be 10 times bigger than the best
                              source_quality_key=None,
                              source_fit_loss_key='no_sparse_fit',
                              gin_variable='ThresholdAnnealer.fit_threshold'):
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
        config[config_object.GIN_KEY][gin_variable] = temp['suggested_hyper']
        temp['suggested_hyper'] = None
        temp['last_hyper_adjustment'] = i
    return config


@gin.configurable
def ThresholdAnnealer(config, epoch_info, temp,
                      fit_threshold=1e-2,
                      min_hyper=1e-5,
                      max_hyper=100,
                      adjust_every=100,
                      reset_on_fail=False,
                      source_metric_key='with_sparse_fit',
                      factor=0.5, **kwargs):
    """Increase sparsity if fit loss is low, decrease otherwise."""

    try:
        fit_loss = find_value(epoch_info, source_metric_key)
        # logging.warning("Cannot find loss with sparsity, defaulting to fit loss")
        logging.info(f"Annealer found loss {fit_loss} {source_metric_key}")
    except AssertionError as e:
        #logging.warning(f"Annealer source metric not found: {source_metric_key}, {e}")
        return config
        # fit_loss = find_value(epoch_info, '/fit/value')

    if 'last_hyper_adjustment' not in temp:
        temp['last_hyper_adjustment'] = 0
    i = epoch_info['epochs']

    if fit_loss > fit_threshold:
        if reset_on_fail:
            temp['suggested_hyper'] = min_hyper
        else:
            if config['losses']['sparsity']['coeff'] > min_hyper:
                temp['suggested_hyper'] = config['losses']['sparsity']['coeff'] * factor
    else:
        if config['losses']['sparsity']['coeff'] < max_hyper:
            temp['suggested_hyper'] = config['losses']['sparsity']['coeff'] / factor

    if temp.get('suggested_hyper', None) is not None and (i - temp['last_hyper_adjustment'] >= adjust_every):
        config['losses']['sparsity']['coeff'] = temp['suggested_hyper']
        temp['suggested_hyper'] = None
        temp['last_hyper_adjustment'] = i
    return config
