import gin
import logging
from sparse_causal_model_learner_rl.config import Config
import sparse_causal_model_learner_rl.loss.losses
import sparse_causal_model_learner_rl.trainable.gumbel_switch as gs
import sparse_causal_model_learner_rl.trainable.fcnet as fcnet
from functools import partial

@gin.configurable
def get_modes():
    return ['no_rotation', 'no_relative',
            'no_mask', 'single_model_dec',
            'single_model_rec', 'ten_features',
            'two_features', 'single_feature',
            'no_model_id_init', 'start_from_zero',
            'original']

@gin.configurable
def process_config(self, mode):
    logging.warning(f"Config ablation mode: {mode}")
    assert mode in get_modes()
    if mode == 'original':
        pass
    elif mode == 'no_rotation':
        gin.bind_parameter('Config.rot_pre', None)
        gin.bind_parameter('Config.rot_post', None)
    elif mode == 'no_relative':
        gin.bind_parameter('fit_loss_obs_space.divide_by_std', False)
    elif mode == 'no_mask':
        gin.bind_parameter('WithInputSwitch.give_mask', False)
    elif mode == 'single_model_dec':
        f_dim = 3
        gin.bind_parameter('ModelDecoder.model_cls', partial(
            fcnet.FCNet, hidden_sizes=[256 * f_dim, 64 * f_dim],
            activation_cls=fcnet.LeakyReLU,
            add_input_batchnorm=True
        ))
    elif mode == 'single_model_rec':
        obs_dim = 35
        gin.bind_parameter('ModelReconstructor.model_cls', partial(
            fcnet.FCNet, hidden_sizes=[256 * obs_dim, 64 * obs_dim],
            activation_cls=fcnet.LeakyReLU,
            add_input_batchnorm=True
        ))
    elif mode == 'ten_features':
        gin.bind_parameter('%N_FEATURES', 10)
    elif mode == 'two_features':
        gin.bind_parameter('%N_FEATURES', 2)
    elif mode == 'single_feature':
        gin.bind_parameter('%N_FEATURES', 1)
    elif mode == 'no_model_id_init':
        gin.bind_parameter('LearnableSwitchSimple.init_identity_up_to', 0)
    elif mode == 'start_from_zero':
        gin.bind_parameter('LearnableSwitchSimple.initial_proba', 0.0)
        gin.bind_parameter('LearnableSwitchSimple.init_identity_up_to', 0)
    else:
        raise ValueError(f"Unknown mode {mode}")