import gin


@gin.configurable
class Config(object):
    """Configurable dictionary that updates itself."""

    # update function key called on update()
    UPDATE_KEY = '_update_function'

    # this key contains variables to be set via gin
    GIN_KEY = '_gin'

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

        # temporary variables for the update function
        self._temporary_variables = {}

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
                gin.bind_parameter(key, value)

    @property
    def config(self):
        return dict(self._config)

    def __repr__(self):
        return f"<Config with keys [{', '.join(self._config.keys())}]>"

    def get(self, *args, **kwargs):
        return self.config.get(*args, **kwargs)

    def __getitem__(self, key):
        return self.config.__getitem__(key)

    def __contains__(self, key):
        return self.config.__contains__(key)