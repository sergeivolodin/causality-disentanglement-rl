# Integration between `tune` and `gin-config`

It is awesome to select hyperparameters automatically with `tune`.
It is cool to specify configuration using `gin-config`.

It is even better to specify parameters for your tune runs via gin!

### Install:
`pip install -e .`

### Usage example
See small_test directory

config.gin
```python
import myfunc
import gin_tune

tune1/grid_search.values = [123, 456]
tune2/grid_search.values = [789, 12]

myfunc.f.x1 = @tune1/grid_search()
myfunc.f.x2 = @tune2/grid_search()

gin_tune_config.num_workers = 0
tune_run.verbose = True
```

myfunc.py
```python
import gin
from ray import tune

@gin.configurable
def f(config, x1, x2):
    """Example function to tune."""
    tune.report(sum=x1+x2)
```

tune.py
```python
from gin_tune import tune_gin
from myfunc import f
import gin


if __name__ == '__main__':
    gin.parse_config_file('config.gin')
    analysis = tune_gin(f)

    print("Sum results")
    print([(x, y['sum']) for x, y in analysis.results.items()])
```

Run `python tune.py`. Should print `[135, 468, 912, 1245]`.

Files `test_config.py` and `test.gin` contain a more complicated example.

### Tests
Just run `pytest`