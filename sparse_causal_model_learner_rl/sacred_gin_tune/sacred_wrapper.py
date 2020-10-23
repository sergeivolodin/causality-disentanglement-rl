from sacred.observers import MongoObserver
from sparse_causal_model_learner_rl.config import Config
import gin
import shutil
import os


def gin_sacred(config_files, main_fcn, db_name='causal_sparse'):
    # creating a sacred experiment
    # https://github.com/IDSIA/sacred/issues/492
    from sacred import Experiment, SETTINGS
    SETTINGS.CONFIG.READ_ONLY_CONFIG = False

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

    name = '_'.join(config_names)
    base_dir = os.getcwd()

    ex = Experiment(name, base_dir=base_dir)
    ex.observers.append(MongoObserver(db_name=db_name))

    for f in config_files:
        if isinstance(f, str):
            f_py = f + '.py'
            shutil.copy(f, f_py)
            ex.add_source_file(f_py)

    config = Config()
    ex.add_config(config=config, **dict(config))

    @ex.main
    def run_train():
        return main_fcn(config=config, ex=ex)

    return ex.run()