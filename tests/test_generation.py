import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from qwen_engine.config import ModelConfig
from qwen_engine.model import QwenModel
from qwen_engine.weights import load_model_weights
from qwen_engine.generation import generate_greedy, generate_greedy_cached, generate_sampled_cached

def test_generation():


    MODEL_PATH = "models/Qwen2.5-0.5B"


    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    text = "The capital of France is"
    max_toks = 10

    input_ids = tokenizer.encode(text, return_tensors="pt")

    config = ModelConfig.load(f"{MODEL_PATH}/config.json")

    model = QwenModel(config)

    load_model_weights(model,f"{MODEL_PATH}/model.safetensors")

    with torch.no_grad():
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


def test_generation_cached():


    MODEL_PATH = "models/Qwen2.5-0.5B"


    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    text = "The capital of France is"
    max_toks = 10

    input_ids = tokenizer.encode(text, return_tensors="pt")

    config = ModelConfig.load(f"{MODEL_PATH}/config.json")

    model = QwenModel(config)

    load_model_weights(model,f"{MODEL_PATH}/model.safetensors")

    with torch.no_grad():
        test_output = generate_greedy(model, input_ids.clone(), max_new_tokens=max_toks)
        cached_output = generate_greedy_cached(model, input_ids.clone(), max_new_tokens=max_toks)

    print("Input:", text)
    print("Test token IDs:", test_output)
    print("Cached run token IDs:", cached_output)
    print("Test output:", tokenizer.decode(test_output[0]))
    print("Cached run output:", tokenizer.decode(cached_output[0]))
    print("Exact token match:", torch.equal(test_output, cached_output))


import torch

from qwen_engine.generation import sample_next_token


def test_sample_1_greedy():
    logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
    sampled = sample_next_token(logits, top_k=1)
    greedy = torch.argmax(logits, dim=-1, keepdim=True)

    assert torch.equal(sampled, greedy)

def test_generation_sampled():

    MODEL_PATH = "models/Qwen2.5-0.5B"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

   # text = "The capital of France is"
    text = '''Answer the following question briefly and directly.
            Question: What is the capital of France?
            Answer: '''
    
    max_toks = 40

    input_ids = tokenizer.encode(text, return_tensors="pt")

    config = ModelConfig.load(f"{MODEL_PATH}/config.json")

    model = QwenModel(config)

    load_model_weights(model,f"{MODEL_PATH}/model.safetensors")

    with torch.no_grad():
        test_output = generate_sampled_cached(model, input_ids.clone(), max_new_tokens=max_toks,
                                              temperature = 0.7, top_k = 5, top_p = 0.95, eos_token_id = tokenizer.eos_token_id)

    print("Input:", text)
    print("Test token IDs:", test_output)
    print("Test output:", tokenizer.decode(test_output[0]))

