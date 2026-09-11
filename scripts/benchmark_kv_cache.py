import time
import torch

from transformers import AutoTokenizer
from qwen_engine.config import ModelConfig
from qwen_engine.model import QwenModel

from qwen_engine.weights import load_model_weights
from qwen_engine.generation import generate_greedy, generate_greedy_cached

# Parameters
torch.set_num_threads(8)
#torch.set_num_interop_threads(
max_toks = 100

MODEL_PATH = "models/Qwen2.5-0.5B"

config = ModelConfig.load(f"{MODEL_PATH}/config.json")

model = QwenModel(config)

load_model_weights(model, f"{MODEL_PATH}/model.safetensors")

model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

input_ids = tokenizer.encode("The capital of France is", return_tensors="pt")



with torch.no_grad():

    start = time.perf_counter()

    no_cache_output = generate_greedy(model, input_ids.clone(), max_toks)

    time1 = time.perf_counter() - start

    model.reset_cache()

    start = time.perf_counter()

    cached_output = generate_greedy_cached(model, input_ids.clone(), max_toks)

    time2 = time.perf_counter() - start


print("Output match:", torch.equal(no_cache_output, cached_output))
print(f"No cache generation time:  {time1:.3f} s")
print(f"Cached generation time: {time2:.3f} s")
print(f"No cache generation:", f"{max_toks / time1:.2f} tps")
print(f"Cached generation: ", f"{max_toks / time2:.2f} tps")
