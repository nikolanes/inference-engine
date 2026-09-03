from safetensors import safe_open

def view_weights(model_path: str) -> None:

    with safe_open(model_path, framework = 'pt', device = 'cpu') as file:
        for key in file.keys():
            tn = file.get_tensor(key)
            print(key, tn.shape, tn.dtype)