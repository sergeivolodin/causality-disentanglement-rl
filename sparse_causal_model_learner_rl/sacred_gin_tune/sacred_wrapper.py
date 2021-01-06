import shutil
import datetime
import logging
import os
import shutil
import sys
import traceback
from functools import partial

import cloudpickle as pickle
import gin
from gin_tune import tune_gin
from matplotlib import pyplot as plt
from path import Path
from ray import tune
from sacred.observers import MongoObserver

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


def main_fcn(config, ex, checkpoint_dir, do_tune=True, do_sacred=True, do_tqdm=False,
             do_exit=True, **kwargs):
    """Main function for gin_sacred."""

    learner_cls = config.get('learner_cls')
    p, m = learner_cls.rsplit('.', 1)
    learner_cls_mod = import_module(p)
    learner_cls = getattr(learner_cls_mod, m)

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

    def callback(self, epoch_info):
        """Callback for Learner."""

        epoch_info = dict(epoch_info)

        # chdir to base_dir
        path_epoch = Path(base_dir) / ("epoch%05d" % self.epochs)

        def add_artifact_local(fn):
            return add_artifact(fn, ex, do_sacred, self.epochs, epoch_info)

        # writing figures if requested
        if self.epochs % self.config.get('graph_every', 5) == 0:
            os.makedirs(path_epoch, exist_ok=True)
            with path_epoch:
                try:
                    threshold, ps, f_out = self.visualize_graph(do_write=True)
                    artifact = path_epoch / (f_out + ".png")
                    add_artifact_local(artifact)
                except Exception as e:
                    logging.error(f"Error plotting causal graph: {self.epochs} {e} {type(e)}")
                    print(traceback.format_exc())

                try:
                    artifact = path_epoch / "threshold_learner_feature.png"
                    add_artifact_local(artifact)
                except Exception as e:
                    logging.error(
                        f"Error plotting threshold for feature: {self.epochs} {e} {type(e)}")
                    print(traceback.format_exc())

                try:
                    artifact = path_epoch / "threshold_learner_action.png"
                    add_artifact_local(artifact)
                except Exception as e:
                    logging.error(
                        f"Error plotting threshold for action: {self.epochs} {e} {type(e)}")
                    print(traceback.format_exc())

                try:
                    fig = self.visualize_model()
                    fig.savefig("model.png", bbox_inches="tight")
                    artifact = path_epoch / "model.png"
                    add_artifact_local(artifact)
                    plt.clf()
                    plt.close(fig)
                except Exception as e:
                    logging.error(f"Error plotting model: {self.epochs} {e} {type(e)}")
                    print(traceback.format_exc())

        if (self.epochs % self.config.get('loss_every', 100) == 0) and self.history:
            os.makedirs(path_epoch, exist_ok=True)
            with path_epoch:
                try:
                    for opt, (fig, ax) in self.visualize_loss_landscape().items():
                        if self._last_loss_mode == '2d':
                            fig.savefig(f"loss_{opt}.png", bbox_inches="tight")
                            artifact = path_epoch / f"loss_{opt}.png"
                            add_artifact_local(artifact)
                            plt.clf()
                            plt.close(fig)
                except Exception as e:
                    logging.error(f"Loss landscape error: {type(e)} {str(e)}")
                    print(traceback.format_exc())

        epoch_info['checkpoint_tune'] = None
        if self.epochs % self.checkpoint_every == 0:
            checkpoint_tune(self, epoch_info)

        # pass metrics to sacred
        if self.epochs % self.config.get('report_every', 1) == 0:
            if do_sacred:
                dict_to_sacred(ex, epoch_info, epoch_info['epochs'])
            if do_tune:
                tune.report(**epoch_info)
            if not do_sacred and not do_tune:
                logging.info(f"Report ready, len={len(epoch_info)}")
        else:
            if do_tune:
                tune.report()

    if checkpoint_dir:
        learner = pickle.load(os.path.join(checkpoint_dir, "checkpoint"))
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
