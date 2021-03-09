import gin
import torch
import logging
from sparse_causal_model_learner_rl.metrics import find_value, find_key

@gin.configurable
def ProjectionThreshold(config, config_object, epoch_info, temp,
        adjust_every=100, metric_threshold=0.5, delta=0.5, source_metric_key=None,
        min_hyper=0, max_hyper=1000,
        gin_variable=None, **kwargs):
    try:
        metric_val = find_value(epoch_info, source_metric_key)
    except AssertionError as e:
        return config

    good = metric_val < metric_threshold
    hyper = gin.query_parameter(gin_variable)
    logging.info(f"Projection: metric={metric_val} threshold={metric_threshold} good={good} hyper={hyper}")

    if 'last_hyper_adjustment' not in temp:
        temp['last_hyper_adjustment'] = 0

    i = epoch_info['epochs']

    if good:
        temp['suggested_hyper'] = hyper - delta
    else:
        temp['suggested_hyper'] = hyper + delta

    if temp['suggested_hyper'] > max_hyper:
        temp['suggested_hyper'] = max_hyper

    if temp['suggested_hyper'] < min_hyper:
        temp['suggested_hyper'] = min_hyper


    if 'suggested_hyper' in temp and (i - temp['last_hyper_adjustment'] >= adjust_every):
        temp['last_hyper_adjustment'] = i
        with gin.unlock_config():
            gin.bind_parameter(gin_variable, temp['suggested_hyper'])
        del temp['suggested_hyper']

    return config
