import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, head_dim: int, rope_theta: float):
        super().__init__()

        # i / (head_dim / 2) = 2i / head_dim
        self.inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

        #self.register_buffer("inv_freq", inv_freq)

        self.cos_cache = None
        self.sin_cache = None
        self.cache_seq_len = 0

    #  simple caching to avoid recomputation
    def extend_cache(self, seq_len: int):
        if seq_len <= self.cache_seq_len:
            return

        # theta(pos, pair_i) = pos / rope_theta ^ (2*pair_i / d)
        positions = torch.arange(self.cache_seq_len, seq_len, device=self.inv_freq.device).float()
        freqs = (positions.reshape(-1, 1) @ self.inv_freq.reshape(1, -1)) 
        #freqs has shape (new_positions_count, head_dim / 2)

        emb = torch.cat((freqs, freqs), dim=-1)
        #emb has shape (new_positions_count, head_dim)
        new_cos = emb.cos()
        new_sin = emb.sin()

        if self.cos_cache is None:
            self.cos_cache = new_cos
            self.sin_cache = new_sin

        else:
            self.cos_cache = torch.cat((self.cos_cache, new_cos), dim=0)
            self.sin_cache = torch.cat((self.sin_cache, new_sin), dim=0)

        self.cache_seq_len = seq_len

    def forward(self, seq_len: int):

        self.extend_cache(seq_len)
        return (self.cos_cache[:seq_len], self.sin_cache[:seq_len])


'''
Qwen2.5 uses half split RoPE instead of consecutive pairs
[x1, x2, x3, x4] -> groups (x1, x3) and (x2, x4)
[x1, x2, x3, x4] -> 
[x1 cos t1 - x3 sin t1, x2 cos t2 - x4 sin t2, x3 cos t1 + x1 sin t1, x4 cos t2 + x2 sin t2]
'''
def reorder_half(x):

    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]

    return torch.cat((-x2, x1), dim = -1)

def apply_rope(x, cos, sin):

    return x * cos + reorder_half(x) * sin