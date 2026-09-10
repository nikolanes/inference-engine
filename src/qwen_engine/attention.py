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

        self.k_cache = None
        self.v_cache = None

    def forward(self, x, cos, sin, use_cache=False):

        batch_size, seq_len, hid_size = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # k,v have shape (batch_size, self.num_kv_heads, seq_len, self.head_dim)
        # k_t has shape (batch_size, self.num_kv_heads, self.head_dim, seq_len)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if use_cache:
            if self.k_cache is None:
                self.k_cache = k
                self.v_cache = v
                
            else:
                self.k_cache = torch.cat((self.k_cache, k), dim=2)
                self.v_cache = torch.cat((self.v_cache, v), dim=2)
                k = self.k_cache
                v = self.v_cache

        prev_len = k.shape[2] - seq_len
        new_len = k.shape[2]

        # for GQA, repeat k_t,v kv_groups (7) times
        k = k.repeat_interleave(self.num_kv_groups, dim=1) 
        v = v.repeat_interleave(self.num_kv_groups, dim=1)
        # k,v now have shape (batch_size, self.num_heads, seq_len, self.head_dim)

        k_t = k.transpose(2,3)

        # logits has shape (batch_size, self.num_heads, seq_len, seq_len)
        logits = q @ k_t
        logits = logits / math.sqrt(self.head_dim)

        #mask = torch.triu(
        #            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), 
        #            diagonal=1)

        mask = torch.triu(torch.ones(seq_len, new_len, device=x.device, dtype=torch.bool),
                          diagonal=prev_len + 1)

        # a token shouldn't look at tokens after it
        logits = logits.masked_fill(mask, float("-inf"))
        softmax = torch.softmax(logits, dim=-1)

        # out has shape (batch_size, self.num_heads, seq_len, self.head_dim)
        out = softmax @ v
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, -1) # merge last 2 dims
        return self.o_proj(out)

    def reset_cache(self):

        self.k_cache = None
        self.v_cache = None

    def cache_len(self):

        return 0 if self.k_cache == None else self.k_cache.shape[2]