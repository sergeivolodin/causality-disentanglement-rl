import gin
import logging


@gin.configurable
def context_rewriter(function, rewrite=None, **kwargs):
    """Change arguments for the function.

    Args:
        function: callable for which to change arguments.
        rewrite: dictionary with rewrite params in format new_key -> old_key
        kwargs: the rest of the arguments (to be rewritten)

    Returns:
        result of function on rewritten arguments.
    """
    assert callable(function), f"Function {function} must be callable"
    if rewrite is None:
        rewrite = {}
        logging.warning("Context rewriter got no arguments")

    kwargs_downstream = dict(kwargs)
    for new_key, old_key in rewrite.items():
        kwargs_downstream[new_key] = kwargs[old_key]

    return function(**kwargs_downstream)