
import torch
from torch import nn

from qwen_engine.attention import Attention
from qwen_engine.layers import MLP, RMSNorm

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):

        add_res = x

        x = self.input_layernorm(x)
        x = self.attention(x, cos, sin)

        x += add_res

        x = self.post_attention_layernorm(x)
        x = self.mlp(x)

        x += add_res
        return x



class QwenModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])

        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
