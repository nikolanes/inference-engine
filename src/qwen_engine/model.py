
import torch
from torch import nn

from qwen_engine.attention import Attention
from qwen_engine.layers import MLP, RMSNorm
from qwen_engine.rope import RoPE

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin, use_cache = False):

        add_res1 = x

        x = self.input_layernorm(x)
        x = self.attention(x, cos, sin, use_cache=use_cache)

        x = x + add_res1
        add_res2 = x

        x = self.post_attention_layernorm(x)
        x = self.mlp(x)

        x = x + add_res2
        return x



class QwenModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for j in range(config.num_hidden_layers)])

        self.rope = RoPE(config.head_dim(), config.rope_theta)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, use_cache = False):

        batch_size, seq_len = input_ids.shape

        prev_len = 0
        if use_cache:
            prev_len = self.layers[0].attention.cache_len()

        x = self.embed_tokens(input_ids)
        # x has shape (batch_size, seq_len, hidden_size)

        cos, sin = self.rope(seq_len, start_pos=prev_len)

        for layer in self.layers:
            x = layer(x, cos, sin, use_cache=use_cache)

        x = self.norm(x)

        # logits has shape (batch_size, seq_len, vocab_size) -> logits[B, i] predicts token i+1
        logits = x @ self.embed_tokens.weight.T
        return logits

    def reset_cache(self):

        for layer in self.layers:
            layer.attention.reset_cache()


