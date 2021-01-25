import gin
import logging
from ray import tune
import ray
from sparse_causal_model_learner_rl.live_parameters.server import run_communicator
from uuid import uuid1
from copy import deepcopy


def param_flatten_dict_keys(dct, separator='//'):
    """Flatten dictionary."""
    flat_dict = {}

    for key, value in dct.items():
        if key.find(separator) >= 0:
            logging.warning(f"Config key {key} contains '{separator}', skipping the key...")
            continue

        if isinstance(value, dict):
            for subkey, subval in param_flatten_dict_keys(value).items():
                flat_dict[f"{key}{separator}{subkey}"] = subval
        else:
            flat_dict[key] = value

    return flat_dict

def param_update_from_flat(dct, update_key, update_value, separator='//'):
    """Update dictionary values based on flattened keys."""
    sep_key = update_key.split(separator)
    if len(sep_key) > 1:
        sk0 = sep_key[0]
        sk_other = sep_key[1:]
        param_update_from_flat(dct[sk0], separator.join(sk_other), update_value)
    else:
        dct[sep_key[0]] = update_value


@gin.configurable
class Config(object):
    """Configurable dictionary that updates itself."""

    # update function key called on update()
    UPDATE_KEY = '_update_function'

    # this key contains variables to be set via gin
    GIN_KEY = '_gin'

    # ignore these keys on pickling
    IGNORE_PICKLE_KEYS = []

    def maybe_start_communicator(self):
        if self.config.get('run_communicator', True):
            name = tune.get_trial_name()
            if name is None:
                name = str(uuid1())
                print(f"Selecting name {name}")
            if not ray.is_initialized():
                ray.init('auto')
            logging.info(f"Starting parameter communicator with name {name}")
            self.communicator = run_communicator(name)

    def update_communicator(self):
        """Set current parameters and update existing ones."""
        if self.communicator is None:
            return

        config_backup = deepcopy(self._config)

        try:
            gin_queries = ray.get(self.communicator.get_clean_gin_queries.remote())

            for q in gin_queries:
                val = gin.query_parameter(q)
                self.communicator.add_msg.remote(f"Gin value {q}: {val}")

            updates = ray.get(self.communicator.get_clear_updates.remote())
            for k, v in updates:
                self.communicator.add_msg.remote(f"Updating parameter {k}: {v}")
                logging.info(f"Updating parameter {k}: {v}")
                param_update_from_flat(self._config, k, v)

            self.communicator.set_current_parameters.remote(param_flatten_dict_keys(self._config))
            self.set_gin_variables()
        except Exception as e:
            self.communicator.add_msg.remote("Error: " + str(e))
            logging.warning(f"Remote parameter update failed {e}")
            self._config = config_backup

    def __getstate__(self):
        to_pickle = {
            '_temporary_variables': self._temporary_variables,
            '_config': {x: y for x, y in self._config.items() if x not in self.IGNORE_PICKLE_KEYS}
        }
        return to_pickle

    def __setstate__(self, s):
        self._config = s['_config']
        self._temporary_variables = s['_temporary_variables']

    def __init__(self, config=None, **kwargs):
        """Initialize

        Args:
            config: dictionary with configuration entries, can be nested
            kwargs: override for config entries.
        """
        if config is None:
            config = {}
        assert isinstance(config, dict), "Please supply a dictionary"
        self._config = config
        self._config.update(kwargs)
        if self.GIN_KEY not in self._config:
            self._config[self.GIN_KEY] = {}

        # temporary variables for the update function
        self._temporary_variables = {}

        self.communicator = None

    def update(self, **kwargs):
        if self.UPDATE_KEY in self._config:
            update_val = self._config[self.UPDATE_KEY]
            if not isinstance(update_val, list):
                update_val = [update_val]
            results = []
            for i, f in enumerate(update_val):
                assert callable(f), f"Update entry {i} must be callable: {f}, {type(f)}"
                results.append(f(config=self._config, temp=self._temporary_variables,
                         **kwargs))
            return results
        return []

    def set_gin_variables(self):
        if self.GIN_KEY in self._config:
            gin_dict = self._config[self.GIN_KEY]
            assert isinstance(gin_dict, dict), f"Gin entry must be a dict {gin_dict}"
            for key, value in gin_dict.items():
                logging.info(f"Binding gin {key} -> {value}")
                gin.bind_parameter(key, value)

    @property
    def config(self):
        return dict(self._config)

    def __repr__(self):
        return f"<Config with keys [{', '.join(self._config.keys())}]>"

    def get(self, *args, **kwargs):
        return self.config.get(*args, **kwargs)

    def set(self, key, val):
        self.config[key] = val

    def __getitem__(self, key):
        return self.config.__getitem__(key)

    def __contains__(self, key):
        return self.config.__contains__(key)

    def __iter__(self):
        for k, v in self.config.items():
            yield k, v