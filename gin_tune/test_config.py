import gin
import gin_tune
import os
from ray import tune
from gin_tune import tune_gin


@gin.configurable
def f(x):
    """A function with one argument."""
    return x

@gin.configurable
def g(x1, x2):
    """A function with two arguments"""
    return (x1, x2)


def test_tune():
    conf_test = os.path.join(gin_tune.__path__[0], 'test.gin')
    gin.parse_config_file(conf_test)

    def fcn(config):
        """Function to run."""
        res = g()
        tune.report(res=res)

    # running tune
    res = tune_gin(fcn)

    # checking results
    res = {x['res'] for x in res.results.values()}

    assert res == {(456, 'caba'), (456, 999), (123, 'caba'), (123, 999)}

    gin.clear_config()