# 6. Problem(decoding)

import torch

def top_p_filter(probs: torch.Tensor, top_p:float) -> torch.Tensor:
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    remove = (cum - sorted_probs) >= top_p
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    filtered = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    return filtered / filtered.sum(dim=-1, keepdim=True)

def sample_next_token(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
) -> torch.Tensor:
    # logits: (V,) 마지막 위치의 점수, 타음 토큰 ID 하나를 돌려줌
    logits = logits.float()
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    probs = top_p_filter(probs, top_p)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    # 프롬프트 뒤에 새로 만든 토큰 ID 리스트를 돌려줌
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    new_ids: list[int] = []
    for _ in range(max_new_tokens):
        x = ids[-context_length:].unsqueeze(0)
        logits = model(x)
        next_id = sample_next_token(logits[0, -1], temperature, top_p, generator)
        tok = int(next_id)
        if eos_id is not None and tok == eos_id:
            break
        new_ids.append(tok)
        ids = torch.cat([ids, next_id.view(1)])
    return new_ids