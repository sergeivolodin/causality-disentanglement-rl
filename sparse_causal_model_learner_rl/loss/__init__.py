losses = {}

def loss(f):
    """Register a function as a loss."""
    losses[f.__name__] = f