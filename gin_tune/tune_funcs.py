import ray.tune as tune
import gin

OVERRIDE_ATTR = '_override'

# map name -> function
FUNCS = {}

def register_func(f):
    """Register tune function."""
    FUNCS[f.__name__] = f
    return f

@gin.configurable
@register_func
def grid_search(values, _override=None):
    return _override

@gin.configurable
@register_func
def choice(categories, _override=None):
    return _override

@gin.configurable
@register_func
def sample_from(func, _override=None):
    return _override