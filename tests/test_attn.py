import torch
from qwen_engine.attention import Attention
from qwen_engine.model import TransformerBlock


class TestConfig:
    hidden_size = 8
    num_attention_heads = 2
    num_key_value_heads = 1
    rms_norm_eps = 1e-6
    intermediate_size = 16

    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

config = TestConfig()

def get_cos_sin(seq_len):

    head_dim = config.head_dim()
    cos = torch.ones(seq_len, head_dim)
    sin = torch.zeros(seq_len, head_dim)
    return cos, sin


def test_attention_shape():
    batch_size = 2
    seq_len = 5

    attn = Attention(config)
    cos, sin = get_cos_sin(seq_len)

    x = torch.randn(batch_size, seq_len, TestConfig.hidden_size)
    out = attn(x, cos, sin)
    assert out.shape == (batch_size, seq_len, TestConfig.hidden_size)

def test_attn_no_nan():
    seq_len = 5

    attn = Attention(config)
    cos, sin = get_cos_sin(seq_len)

    x = torch.randn(2, seq_len, TestConfig.hidden_size)
    out = attn(x, cos, sin)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

def test_attn_causal():

    seq_len = 4

    attn = Attention(config)
    cos, sin = get_cos_sin(seq_len)

    x1 = torch.randn(1, seq_len, TestConfig.hidden_size)

    x2 = x1.clone()

    x2[:, 3, :] = torch.randn(TestConfig.hidden_size) * 100

    out1 = attn(x1, cos, sin)
    out2 = attn(x2, cos, sin)

    assert torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-6)


def test_transformer_block():
    
    block = TransformerBlock(config)

    batch_size = 2
    seq_len = 4

    x = torch.randn(batch_size, seq_len, config.hidden_size)

    cos, sin = get_cos_sin(seq_len)

    out = block(x, cos, sin)

    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()