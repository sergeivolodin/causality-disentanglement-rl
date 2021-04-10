import numpy as np
import torch
from torch import nn
import gin


@gin.configurable
class Quadratic(nn.Module):
    """Quadratic layer. y_i=x^TW_ix + w_i^Tx + b_i."""
    def __init__(self, in_features=None, out_features=None):
        super(Quadratic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        Winit = np.random.randn(out_features, in_features, in_features).astype(np.float32) / 100
        winit = np.random.randn(out_features, in_features).astype(np.float32)
        binit = np.zeros(out_features, dtype=np.float32)
        
        self.W = nn.Parameter(torch.from_numpy(Winit), requires_grad=True)
        self.w = nn.Parameter(torch.from_numpy(winit), requires_grad=True)
        self.b = nn.Parameter(torch.from_numpy(binit), requires_grad=True)
    def forward(self, x):
        out = torch.einsum('bf,gf->bg', x, self.w)
        out += torch.einsum('bf,be,gfe->bg', x, x, self.W)
        out += self.b.view(1, self.b.shape[0])
        return out
    
    def __repr__(self):
        return f"<Quadratic in_f={self.in_features} out_f={self.out_features}>"
