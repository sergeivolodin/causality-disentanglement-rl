from sparse_causal_model_learner_rl.sacred_gin_tune.sacred_wrapper import load_config_files
import gin
import tempfile
import pytest
import os
from ray import tune


@gin.configurable
def f(param):
    return param

@gin.configurable
def g(param1, param2):
    return (param1, param2)

@pytest.fixture
def f_conf_123():
    conf_file = """
        import sparse_causal_model_learner_rl.sacred_gin_tune.test_load_gin_file
        test_load_gin_file.f.param = 123
        """
    return conf_file


def test_load_file(f_conf_123):
    with tempfile.NamedTemporaryFile(mode='w+') as tmp_config_file:
        tmp_config = tmp_config_file.name
        tmp_config_file.write(f_conf_123)
        tmp_config_file.flush()

        gin.parse_config_file(tmp_config)
    assert f() == 123
    gin.clear_config()


def test_load_file_via_fcn(f_conf_123):
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_config_file:
        tmp_config = tmp_config_file.name
        tmp_config_file.write(f_conf_123)

    load_config_files([tmp_config])
    assert f() == 123
    os.unlink(tmp_config)
    gin.clear_config()
