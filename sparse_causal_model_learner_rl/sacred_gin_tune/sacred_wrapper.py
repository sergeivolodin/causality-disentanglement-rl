import os
import shutil
from functools import partial
import datetime

import gin
from sacred.observers import MongoObserver

from gin_tune import tune_gin
from sparse_causal_model_learner_rl.config import Config


def load_config_files(config_files):
    """Load .gin config files into the python environment."""
    config_names = []
    for c in config_files:
        if callable(c):
            c()
            config_names.append(str(c))
        elif isinstance(c, str):
            gin.parse_config_file(c)
            config_names.append(os.path.basename(c)[:-4])
        else:
            raise TypeError(f"Config file can be either a callable or a string: {c}")
    return config_names


def sacred_experiment_with_config(config, name, main_fcn, db_name, base_dir, checkpoint_dir, sources=[]):
    """Launch a sacred experiment."""
    # creating a sacred experiment
    # https://github.com/IDSIA/sacred/issues/492
    from sacred import Experiment, SETTINGS
    SETTINGS.CONFIG.READ_ONLY_CONFIG = False

    ex = Experiment(name, base_dir=base_dir)
    ex.observers.append(MongoObserver(db_name=db_name))

    for f in sources:
        if isinstance(f, str):
            f_py = f + '.py'
            shutil.copy(f, f_py)
            ex.add_source_file(f_py)

    ex.add_config(config=config, **dict(config))

    @ex.main
    def run_train():
        return main_fcn(config=config, checkpoint_dir=checkpoint_dir, ex=ex)

    return ex.run()


def inner_fcn(config, main_fcn=None, checkpoint_dir=None, **kwargs):
    """Inner fuction running inside tune."""

    kwargs = dict(main_fcn=main_fcn, name=config['name'],
                  base_dir=config['base_dir'], db_name=config['db_name'],
                  sources=config['sources'], checkpoint_dir=checkpoint_dir)

    config_ = Config()

    result = sacred_experiment_with_config(config=config_, **kwargs)
    return None


def gin_sacred(config_files, main_fcn, db_name='causal_sparse', base_dir=None):
    """launch a sacred experiment from .gin config files."""
    config_names = load_config_files(config_files)

    name = '_'.join(config_names)
    if base_dir is None:
        base_dir = os.getcwd()

    run_uid = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")

    base_dir = os.path.join(base_dir, name, run_uid)

    os.makedirs(base_dir, exist_ok=True)

    inner_fcn1 = partial(inner_fcn, main_fcn=main_fcn)
    inner_fcn1.__name__ = main_fcn.__name__

    analysis = tune_gin(inner_fcn1, config_update={'name': name, 'base_dir': base_dir,
                                                   'db_name': db_name, 'sources': config_files},
                        name=name)

    return analysis
