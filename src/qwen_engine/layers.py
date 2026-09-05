import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(self, hidden_size : int, epsilon : float):
        super().__init__();

        self.epsilon = epsilon
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x) -> int:
        mean_sq = x.pow(2).mean(dim = -1, keepdim = True)
        rms = torch.sqrt(mean_sq + self.epsilon)
        norm = (x / rms) * self.weight
        return norm
        

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):

        gate = self.gate_proj(x)
        up = self.up_proj(x)

        x_new = torch.nn.functional.silu(gate) * up

        return self.down_proj(x_new)
