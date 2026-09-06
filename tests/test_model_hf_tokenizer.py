import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_engine.config import ModelConfig
from qwen_engine.model import QwenModel
from qwen_engine.weights import load_model_weights

def test_model_hf_tokenizer():


    MODEL_PATH = "models/Qwen2.5-0.5B"

    config = ModelConfig.load(f"{MODEL_PATH}/config.json")
    model = QwenModel(config)
    load_model_weights(model, f"{MODEL_PATH}/model.safetensors")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    text = "Hello world!"

    input_ids = tokenizer.encode(text, return_tensors="pt")

    print("\n\nTokenizer total size:", len(tokenizer), "\nTokenizer vocab size:", tokenizer.vocab_size)
    print("\nText:", text, "\nToken ids:", input_ids, "\nInput size:", input_ids.shape)
    print(tokenizer.convert_ids_to_tokens([9707, 1879, 0]))
    with torch.no_grad():
        logits = model(input_ids)

    print("\nLogits shape:", logits.shape)

    last_tok_logits = logits[0, -1]
    next_tok_id = torch.argmax(last_tok_logits).item()

    print("\nMy predicted next token ID:", next_tok_id)
    print("My predicted next token:", tokenizer.decode([next_tok_id]))


    hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32)

    with torch.no_grad():
        hf_log = hf_model(input_ids).logits

    next_id = torch.argmax(hf_log[0, -1]).item()

    print("HF token ID:", next_id)
    print("HF token:", repr(tokenizer.decode([next_id])))

    print("Max logit error:", torch.max(torch.abs(logits - hf_log)).item())