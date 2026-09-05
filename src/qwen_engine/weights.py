from safetensors import safe_open
import torch

def view_weights(model_path: str) -> None:

    with safe_open(model_path, framework = 'pt', device = 'cpu') as file:
        for key in file.keys():
            tn = file.get_tensor(key)
            print(key, tn.shape, tn.dtype)

def attention_load_weights(attn, model_path : str, layer_index : int) -> None:

    name = f"model.layers.{layer_index}.self_attn"
    with safe_open(model_path, framework = 'pt', device = 'cpu') as file:

        with torch.no_grad():
            
            attn.q_proj.weight.copy_(file.get_tensor(f"prefix.q_proj.weight"))
            attn.q_proj.bias.copy_(file.get_tensor(f"prefix.q_proj.bias"))

            attn.k_proj.weight.copy_(file.get_tensor(f"prefix.k_proj.weight"))
            attn.k_proj.bias.copy_(file.get_tensor(f"prefix.k_proj.bias"))

            attn.q_proj.weight.copy_(file.get_tensor(f"prefix.v_proj.weight"))
            attn.q_proj.bias.copy_(file.get_tensor(f"prefix.v_proj.bias"))

            attn.o_proj.weight.copy_(file.get_tensor(f"prefix.o_proj.weight"))

def load_block_weights(file, block, layer_idx):
    prefix = f"model.layers.{layer_idx}"

    with torch.no_grad():

        block.input_layernorm.weight.copy_(file.get_tensor(f"{prefix}.input_layernorm.weight"))
        block.post_attention_layernorm.weight.copy_(file.get_tensor(f"{prefix}.post_attention_layernorm.weight"))

        block.attention.q_proj.weight.copy_(file.get_tensor(f"{prefix}.self_attn.q_proj.weight"))
        block.attention.q_proj.bias.copy_(file.get_tensor(f"{prefix}.self_attn.q_proj.bias"))
        block.attention.k_proj.weight.copy_(file.get_tensor(f"{prefix}.self_attn.k_proj.weight"))
        block.attention.k_proj.bias.copy_(file.get_tensor(f"{prefix}.self_attn.k_proj.bias"))
        block.attention.v_proj.weight.copy_(file.get_tensor(f"{prefix}.self_attn.v_proj.weight"))
        block.attention.v_proj.bias.copy_(file.get_tensor(f"{prefix}.self_attn.v_proj.bias"))
        block.attention.o_proj.weight.copy_(file.get_tensor(f"{prefix}.self_attn.o_proj.weight"))
        
        block.mlp.gate_proj.weight.copy_(file.get_tensor(f"{prefix}.mlp.gate_proj.weight"))
        block.mlp.up_proj.weight.copy_(file.get_tensor(f"{prefix}.mlp.up_proj.weight"))
        block.mlp.down_proj.weight.copy_(file.get_tensor(f"{prefix}.mlp.down_proj.weight"))