import torch

class WeightRestorer(object):
    """Save weights for all given models, restore on exit."""
    def __init__(self, models):
        assert isinstance(models, list), f"Please supply a list of models, got {type(models)}"
        for model in models:
            assert isinstance(model, torch.nn.Module), f"Please supply a list of models, got {type(model)}"
        self.models = models
        self.saved = []

    def __enter__(self):
        self.saved = []
        for m in self.models:
            ckpt = m.state_dict()
            self.saved.append(ckpt)

    def __exit__(self, type, value, traceback):
        for m, ckpt in zip(self.models, self.saved):
            m.load_state_dict(ckpt)
        self.saved = []