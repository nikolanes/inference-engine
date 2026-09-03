import torch
import torch.nn as nn

from qwen_engine.layers import RMSNorm


def test_rmsnorm():
    hidden_size = 8
    eps = 1e-6

    test = RMSNorm(hidden_size, eps)
    torch_rms = nn.RMSNorm(hidden_size, eps=eps)
    torch_rms.weight.data.copy_(test.weight.data)

    x = torch.randn(2, 3, hidden_size)

    test_out = test(x)
    torch_out = torch_rms(x)

    assert torch.allclose(test_out, torch_out, atol=1e-6)
    print("\nMaximum abs difference RMSNorm:", torch.max(torch.abs(test_out - torch_out)).item())