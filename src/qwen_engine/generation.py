import torch


def generate_greedy(model, input_ids, max_new_tokens):

    with torch.no_grad():

        model.eval()

        for i in range(max_new_tokens):
            logits = model(input_ids)

            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            input_ids = torch.cat((input_ids, next_tok), dim=1)

        return input_ids