import torch


def generate_greedy(model, input_ids, max_new_tokens):

    model.eval()

    for i in range(max_new_tokens):
        logits = model(input_ids)

        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        input_ids = torch.cat((input_ids, next_tok), dim=1)

    return input_ids

def generate_greedy_cached(model, input_ids, max_new_tokens):


    model.eval()
    model.reset_cache()

    curr_toks = input_ids.clone()

    logits = model(input_ids, use_cache=True)

    for i in range(max_new_tokens):

        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        curr_toks = torch.cat((curr_toks, next_tok), dim=1)

        if (i+1 == max_new_tokens):
            break

        logits = model(next_tok, use_cache=True)

    return curr_toks

import torch


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):


    if temperature <= 0:
        raise ValueError("Temperature parameter must be > 0")

    if top_k is not None and top_k <= 0:
        raise ValueError("Top-k parameter must be >= 1")

    if top_p is not None and not (0 < top_p <= 1):
        raise ValueError("Top-p parameter must be in (0, 1]")

    logits = logits / temperature
    idx = None

    if top_k is not None:

        top_k = min(top_k, logits.shape[-1])
        values, idx = torch.topk(logits, top_k)
        #top_kth = values[:, -1].reshape(-1, 1)
        logits = values

    if top_p is not None:

        log_sorted, idx_sorted = torch.sort(logits, descending=True, dim=-1)
        prob = torch.softmax(log_sorted, dim=-1)
        prob_prefix_sum = torch.cumsum(prob, dim=-1)

        remove = prob_prefix_sum > top_p

        remove[..., 1:] = remove[..., :-1].clone() # hack to convert [0,0,0,1,1] -> [0,0,0,0,1]
        remove[..., 0] = False; # in case remove[..., 0] is 1 (prob[..., 0] > top_p)

        log_sorted = log_sorted.masked_fill(remove, float("-inf"))

        final_probs = torch.softmax(log_sorted, dim=-1)
        tok_sorted = torch.multinomial(final_probs, num_samples=1)

        next_token = idx_sorted.gather(dim=-1, index=tok_sorted)

        if (idx is not None):
            next_token = idx.gather(dim=-1, index=next_token)

        return next_token

    # no top-p
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    if idx is not None:
        next_token = idx.gather(dim=-1, index=next_token)
    return next_token


def generate_sampled_cached(model, input_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_token_id=None):

    model.eval()
    model.reset_cache()
    output = input_ids.clone()

    logits = model(input_ids, use_cache=True)

    for i in range(max_new_tokens):

        next_token = sample_next_token(logits[..., -1, :], temperature=temperature, top_k=top_k, top_p=top_p)
        output = torch.cat((output, next_token), dim=1)

        if next_token.item() == eos_token_id:
            break

        if (i+1 == max_new_tokens):
            break
        logits = model(next_token, use_cache=True)

    return output