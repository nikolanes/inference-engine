import math
import torch
import torch.nn as nn
from qwen_engine.rope import apply_rope


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim()

        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim,bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim,bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x, cos, sin):

        batch_size, seq_len, hid_size = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # k,v have shape (batch_size, self.num_kv_heads, seq_len, self.head_dim)
        # k_t has shape (batch_size, self.num_kv_heads, self.head_dim, seq_len)

        q = apply_rope(q, self.cos, self.sin)
        k = apply_rope(k, self.cos, self.sin)

        # for GQA, repeat k_t,v kv_groups (7) times
        k = k.repeat_interleave(self.num_kv_groups, dim=1) 
        v = v.repeat_interleave(self.num_kv_groups, dim=1)
        # k,v now have shape (batch_size, self.num_heads, seq_len, self.head_dim)

        k_t = k.transpose(2,3)

        # logits has shape (batch_size, self.num_heads, seq_len, seq_len)
        logits = q @ k_t
        logits = logits / math.sqrt(self.head_dim)

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), 
            diagonal=1)

        logits = logits.masked_fill(mask, float("-inf"))
        softmax = torch.softmax(logits, dim=-1)

        # out has shape (batch_size, self.num_heads, seq_len, self.head_dim)
        out = softmax @ v
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, -1) # merge last 2 dims
        return self.o_proj(out)