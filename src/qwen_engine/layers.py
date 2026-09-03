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
        

