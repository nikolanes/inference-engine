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
        logits = model(next_tok, use_cache=True)

    return curr_toks