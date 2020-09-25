import gin
from sparse_causal_model_learner_rl.config import Config


@gin.configurable
def gin_function(x):
    return x


def test_config_create():
    c = Config()
    assert c.config == {}


def test_config_get():
    c = Config(config={'a': 123})
    assert c.config['a'] == 123


def test_config_override():
    c = Config(config={'a': 123}, a=456)
    assert c.config['a'] == 456


def test_config_gin():
    c = Config(config={Config.GIN_KEY: {'gin_function.x': 123}})
    c.set_gin_variables()
    assert gin_function() == 123


def test_config_update():
    def update_function(config, temp, next_a_val, **kwargs):
        config['a'] = next_a_val

    c = Config(config={Config.UPDATE_KEY: update_function, 'a': 123})
    assert c.config['a'] == 123
    c.update(next_a_val=456)
    assert c.config['a'] == 456
