from sacred.observers import MongoObserver
from sparse_causal_model_learner_rl.config import Config
import gin
import shutil
import os


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

def sacred_experiment_with_config(config, main_fcn, db_name, base_dir, sources=[]):
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
        return main_fcn(config=config, ex=ex)

    return ex.run()



def gin_sacred(config_files, main_fcn, db_name='causal_sparse'):
    """launch a sacred experiment from .gin config files."""
    config_names = load_config_files(config_files)

    name = '_'.join(config_names)
    base_dir = os.getcwd()

    result = sacred_experiment_with_config(config=Config(), main_fcn=main_fcn, name=name,
                                           base_dir=base_dir, db_name=db_name, sources=config_files)

    return result