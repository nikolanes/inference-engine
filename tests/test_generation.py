import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from qwen_engine.config import ModelConfig
from qwen_engine.model import QwenModel
from qwen_engine.weights import load_model_weights
from qwen_engine.generation import generate_greedy

def test_generation():


    MODEL_PATH = "models/Qwen2.5-0.5B"


    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    text = "The capital of France is"
    max_toks = 10

    input_ids = tokenizer.encode(text, return_tensors="pt")

    config = ModelConfig.load(f"{MODEL_PATH}/config.json")

    model = QwenModel(config)

    load_model_weights(model,f"{MODEL_PATH}/model.safetensors")

    my_output = generate_greedy(model, input_ids.clone(), max_new_tokens=max_toks)

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)

    hf_model.eval()

    with torch.no_grad():
        hf_output = hf_model.generate(input_ids.clone(), max_new_tokens=max_toks, do_sample=False)

    print("Input:", text)
    print("My model token IDs:", my_output)
    print("HF token IDs:", hf_output)
    print("My model output:", tokenizer.decode(my_output[0]))
    print("HF output:", tokenizer.decode(hf_output[0]))
    print("Exact token match:", torch.equal(my_output, hf_output))