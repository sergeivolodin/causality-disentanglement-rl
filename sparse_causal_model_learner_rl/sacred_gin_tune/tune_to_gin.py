import ray.tune as tune
import gin

@gin.configurable
def tune_grid_search(values, override=None):
    """Grid search, gin-configurable."""
    return override

@gin.configurable
def tune_choice(values, override=None):
    return override
