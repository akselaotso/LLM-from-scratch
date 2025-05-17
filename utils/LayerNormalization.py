import torch.nn as nn, torch

class LayerNormalization(nn.Module):
    def __init__(self, dimension):
        super().__init__()

        self.scale = nn.Parameter(torch.ones(dimension))
        self.shift = nn.Parameter(torch.zeros(dimension))

        
    def forward(self, x: torch.tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)

        return self.scale * x_norm + self.shift
    

