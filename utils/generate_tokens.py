import torch

def generate_tokens(model, encoded_tensor, max_tokens, context_size, temperature=0.0, top_k=0, eos_id=None):
    for _ in range(max_tokens):
        current_context = encoded_tensor[:, -context_size:]

        with torch.no_grad():
            model.eval()
            logits = model(current_context)

        logits = logits[:, -1, :]

        if top_k > 0:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            logits = torch.softmax(logits, dim=-1)
            logit = torch.multinomial(logits, num_samples=1)
        else:
            logit = torch.argmax(logits, dim=-1, keepdim=True)

        if logit == eos_id:
           break

        encoded_tensor = torch.cat((encoded_tensor, logit), dim=-1)

    return encoded_tensor

