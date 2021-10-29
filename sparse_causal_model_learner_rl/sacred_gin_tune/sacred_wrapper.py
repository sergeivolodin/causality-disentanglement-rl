import shutil
import datetime
import logging
import os
import shutil
import sys
from functools import partial
from uuid import uuid1

import cloudpickle as pickle
import gin
from gin_tune import tune_gin
from path import Path
from ray import tune
from sacred.observers import QueuedMongoObserver

from causal_util.helpers import dict_to_sacred
from sparse_causal_model_learner_rl.config import Config
from sparse_causal_model_learner_rl.visual.learner_visual import add_artifact
import matplotlib as mpl
from importlib import import_module


def load_config_files(config_files):
    """Load .gin config files into the python environment."""
    config_names = []
    for c in config_files:
        if callable(c):
            c()
            config_names.append(str(c))
        elif isinstance(c, str):
            c = os.path.abspath(c)
            with Path(os.path.dirname(c)):
                gin.parse_config_file(c)
            config_names.append(os.path.basename(c)[:-4])
        else:
            raise TypeError(f"Config file can be either a callable or a string: {c}")
    return config_names


def sacred_experiment_with_config(config, name, main_fcn, db_name, base_dir, checkpoint_dir, sources=[], tune_config={}):
    """Launch a sacred experiment."""
    # creating a sacred experiment
    # https://github.com/IDSIA/sacred/issues/492
    from sacred import Experiment, SETTINGS
    SETTINGS.CONFIG.READ_ONLY_CONFIG = False

    ex = Experiment(name, base_dir=base_dir)
    ex.observers.append(QueuedMongoObserver(db_name=db_name))

    for f in sources:
        if isinstance(f, str):
            f_py = f + '.py'
            shutil.copy(f, f_py)
            ex.add_source_file(f_py)

    export_config = dict(config)
    export_config.update(tune_config)
    ex.add_config(config=tune_config, **tune_config)

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

    result = sacred_experiment_with_config(config=config_, tune_config=config, **kwargs)
    return None


def gin_sacred(config_files, main_fcn, db_name='causal_sparse', base_dir=None):
    """launch a sacred experiment from .gin config files."""
    config_names = load_config_files(config_files)
    gin.finalize()

    name = '_'.join(config_names)
    if base_dir is None:
        base_dir = os.getcwd()

    run_uid = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")
    run_uid += "__" + str(uuid1())

    base_dir = os.path.join(base_dir, name, run_uid)

    os.makedirs(base_dir, exist_ok=True)

    inner_fcn1 = partial(inner_fcn, main_fcn=main_fcn)
    inner_fcn1.__name__ = main_fcn.__name__

    analysis = tune_gin(inner_fcn1, config_update={'name': name, 'base_dir': base_dir,
                                                   'db_name': db_name, 'sources': config_files},
                        name=name)

    return analysis


def main_fcn(config, ex, checkpoint_dir, do_tune=True, do_sacred=True, do_tqdm=False,
             do_exit=True, **kwargs):
    """Main function for gin_sacred."""

    learner_cls = config.get('learner_cls')
    p, m = learner_cls.rsplit('.', 1)
    learner_cls_mod = import_module(p)
    learner_cls = getattr(learner_cls_mod, m)

    if config.get('load_checkpoint', None) is not None:
        checkpoint_dir = config.get('load_checkpoint')

    if do_sacred:
        base_dir = ex.base_dir
    else:
        base_dir = '/tmp/'

    def checkpoint_tune(self, epoch_info=None):
        """Checkpoint, possibly with tune."""
        if epoch_info is None:
            epoch_info = self.epoch_info

        if do_tune:
            with tune.checkpoint_dir(step=self.epochs) as checkpoint_dir:
                ckpt = self.checkpoint(checkpoint_dir)
                epoch_info['checkpoint_tune'] = ckpt
                epoch_info['checkpoint_size'] = os.path.getsize(ckpt)
        else:
            ckpt_dir = os.path.join(base_dir, "checkpoint%05d" % epoch_info['epochs'])
            os.makedirs(ckpt_dir, exist_ok=True)
            self.checkpoint(ckpt_dir)
            logging.info(f"Checkpoint available: {ckpt_dir}")

    mpl.use('Agg')

    def callback(self, epoch_info, epoch_profiler):
        """Callback for Learner."""

        epoch_profiler.start('callback_pre')

        epoch_info = dict(epoch_info)

        # chdir to base_dir
        path_epoch = Path(base_dir) / ("epoch%05d" % self.epochs)

        def add_artifact_local(fn):
            if not os.path.isfile(fn):
                logging.warning("Can't add artifact because file does not exist " + fn)
                return
            return add_artifact(fn, ex, do_sacred, self.epochs, epoch_info)

        epoch_profiler.end('callback_pre')
        epoch_profiler.start('artifacts')
        self.maybe_write_artifacts(path_epoch, add_artifact_local)
        epoch_profiler.end('artifacts')

        epoch_profiler.start('checkpoint')
        epoch_info['checkpoint_tune'] = None
        if self.epochs % self.checkpoint_every == 0:
            checkpoint_tune(self, epoch_info)
        epoch_profiler.end('checkpoint')

        # pass metrics to tune
        epoch_profiler.start('report_tune')
        if self.epochs % self.config.get('report_every', 1) == 0:
            if do_tune:
                epoch_profiler.start('report_tune_data')
                tune.report(**epoch_info)
                epoch_profiler.end('report_tune_data')
            if not do_sacred and not do_tune:
                logging.info(f"Report ready, len={len(epoch_info)}")
        else:
            if do_tune and not self.config.get('tune_no_empty_report', False):
                epoch_profiler.start('report_tune_empty')
                tune.report()
                epoch_profiler.end('report_tune_empty')
        epoch_profiler.end('report_tune')
        epoch_profiler.start('report_sacred')
        if do_sacred and self.epochs % self.config.get('sacred_every', 1) == 0:
            dict_to_sacred(ex, epoch_info, epoch_info['epochs'])
        epoch_profiler.end('report_sacred')

    if checkpoint_dir:
        learner = pickle.load(open(os.path.join(checkpoint_dir, "checkpoint"), 'rb'))
        learner.callback = callback
    else:
        learner = learner_cls(config, callback=callback)

    learner.train(do_tqdm=do_tqdm)

    # last checkpoint at the end
    checkpoint_tune(learner)

    # closing all resources
    del learner

    if do_exit:
        sys.exit(0)

    return None


def learner_gin_sacred(configs, nofail=False):
    """Launch Learner from gin configs."""
    main_fcn_use = main_fcn
    if nofail:
        main_fcn_use = partial(main_fcn_use, do_exit=False)
    # load_config_files(configs)
    main_fcn_use.__name__ = "main_fcn"
    return gin_sacred(configs, main_fcn_use, db_name='causal_sparse',
                      base_dir=os.path.join(os.getcwd(), 'results'))
