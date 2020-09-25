import gin


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
            f = self._config[self.UPDATE_KEY]
            assert callable(f), f"Update entry must be callable {f}"
            return f(config=self._config, temp=self._temporary_variables,
                     **kwargs)

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
        return f"<Config with keys {self._config.keys()}"
