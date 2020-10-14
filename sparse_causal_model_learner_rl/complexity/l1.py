import torch

from sparse_causal_model_learner_rl.complexity.complexity_metric import ComplexityMetric


class L1(ComplexityMetric):
    def __init__(self):
        super(L1, self).__init__()

    def __call__(self, w):
        assert isinstance(w, torch.Tensor), f"Please supply a tensor {w}"
        return torch.sum(torch.abs(w))


class Lp(ComplexityMetric):
    def __init__(self, p):
        assert isinstance(p, float), f"Please give a float parameter p {p}"
        super(Lp, self).__init__(p=p)

    def __call__(self, w):
        assert isinstance(w, torch.Tensor), f"Please supply a tensor {w}"
        return torch.norm(w.flatten(), p=self.params['p'])
