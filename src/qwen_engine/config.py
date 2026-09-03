import json
from dataclasses import dataclass

@dataclass
class ModelConfig:

    vocab_size: int
    hidden_size: int
    intermediate_size: int

    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int

    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int

    tie_word_embeddings: bool

    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def load(cls, config_path):
        with open(config_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        return cls(vocab_size = data['vocab_size'],
        hidden_size = data['hidden_size'],
        intermediate_size = data['intermediate_size'],

        num_hidden_layers = data['num_hidden_layers'],
        num_attention_heads = data['num_attention_heads'],
        num_key_value_heads = data['num_key_value_heads'],
        #head_dim = hidden_size // num_attention_heads,
        rms_norm_eps = data['rms_norm_eps'],
        rope_theta = data['rope_theta'],
        max_position_embeddings = data['max_position_embeddings'],
        tie_word_embeddings = data['tie_word_embeddings'])
