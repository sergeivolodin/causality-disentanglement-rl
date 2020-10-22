import gin


def find_key(dct, key_substr):
    """Get a key from a dictionary containing a substring."""
    keys = dct.keys()
    keys_match = [k for k in keys if key_substr in k]
    assert len(keys_match) == 1, f"Unknown substring {key_substr} in {keys}, appears {len(keys_match)} times"
    return keys_match[0]


def find_value(dct, key_substr):
    """Get a value from a dict given a substring in a key."""
    return dct[find_key(dct, key_substr)]


@gin.configurable
def ThresholdAnnealer(config, epoch_info, temp,
                      fit_threshold=1e-2,
                      min_hyper=1e-5,
                      max_hyper=10,
                      factor=0.5, **kwargs):
    """Increase sparsity if fit loss is low, decrease otherwise."""
    rec_loss = find_value(epoch_info, '/reconstruction/value')
    fit_loss = find_value(epoch_info, '/fit/value')

    if 'last_hyper_adjustment' not in temp:
        temp['last_hyper_adjustment'] = 0
    i = epoch_info['epochs']

    if rec_loss + fit_loss > fit_threshold:
        if config['losses']['sparsity']['coeff'] > min_hyper:
            suggested_hyper = config['losses']['sparsity']['coeff'] * factor
    else:
        if config['losses']['sparsity']['coeff'] < max_hyper:
            suggested_hyper = config['losses']['sparsity']['coeff'] / factor

    if i - temp['last_hyper_adjustment'] >= 100:
        config['losses']['sparsity']['coeff'] = suggested_hyper
        temp['last_hyper_adjustment'] = i
    return config
