"""11/24 kv 캐시를 쓰는 생성"""

import torch
from einops import rearrange

from cs336_basics.decoding import sample_next_token
from cs336_basics.model import scaled_dot_product_attention

class KVcache:
    """층마다 k, v를 (B, h, max_len, d_head) 크기로 미리 잡아두고, length 포인터로 다음 칸에 쓰기"""
    def __init__(self, num_layers, batch, num_heads, max_len, d_head, device, dtype):
        shape = (batch, num_heads, max_len, d_head)
        # 층 수만큼 빈 텐서 만들기.
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.max_len = max_len
        self.length = 0     # 지금까지 채운 칸 수 = 다음 토큰이 들어갈 위치

    @classmethod
    def for_model(cls, model, batch: int = 1, max_len: int | None = None):
        # 모델 모양(층 수, 헤드 수, d_head, 장치, 정밀도)를 읽어 맞는 캐시 만들기
        attn = model.layers[0].attn
        p = next(model.parameters())
        return cls(
            num_layers=len(model.layers),
            batch=batch,
            num_heads=attn.num_heads,
            max_len=max_len or model.context_length,
            d_head=attn.d_head,
            device=p.device,
            dtype=p.dtype,
        )

    def nbytes(self) -> int:
        # 미리 잡아 둔 캐시 전체 크기(바이트)
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

def attention_with_cache(attn, x, cache: KVcache, layer_idx: int, start: int) -> torch.Tensor:
    """MultiheadselfAttention.forward의 캐시 버전
    
    x: (B, I, d_model) I는 이번에 새로 넣는 토큰 수 (prefill이면 P, decode면 1)
    start: 이번 토큰들의 첫 위치 (캐시에 이미 있는 칸 수)"""

    I = x.shape[-2]
    end = start + I

    # 1) 새 토큰들의 q, k, v 생성
    q = rearrange(attn.q_proj(x), "... seq (h d) -> ... h seq d", h=attn.num_heads)
    k = rearrange(attn.k_proj(x), "... seq (h d) -> ... h seq d", h=attn.num_heads)
    v = rearrange(attn.v_proj(x), "... seq (h d) -> ... h seq d", h=attn.num_heads)

    # 2) RoPE 위치는 0부터가 아니라 start 부터
    if attn.rope is not None:
        pos = torch.arange(start, end, device=x.device)
        q = attn.rope(q, pos)
        k = attn.rope(k, pos)

    # 3) RoPE를 적용한 K와 v를 캐시 start ~ end-1칸에 써 넣기
    cache.k[layer_idx][:, :, start:end] = k
    cache.v[layer_idx][:, :, start:end] = v

    # 4) 캐시에서 0 ~ end-1 칸 읽기
    k = cache.k[layer_idx][:, :, :end]
    v = cache.v[layer_idx][:, :, :end]

    # 5) causal mask: 이번 query i는 key j<=start+i만 본다
    # decode(I=1)는 마지막 토큰이라 전부 볼 수 있으므로 마스크가 필요 업음
    if I == 1:
        mask = None
    else:
        q_pos = torch.arange(start, end, device=x.device).unsqueeze(-1)     #(I, 1)
        k_pos = torch.arange(end, device=x.device).unsqueeze(0)             #(1, end)
        mask = k_pos<=q_pos                                                 #(I, end)

    out = scaled_dot_product_attention(q, k, v, mask)
    out = rearrange(out, "... h seq d -> ... seq (h d)")
    return attn.output_proj(out)

def block_with_cache(block, x, cache:KVcache, layer_idx: int, start: int) -> torch.Tensor:
    """TransformerBlock.forward의 캐시 버전. attention만 바뀌고 나머지는 원래 그대로"""
    x = x + attention_with_cache(block.attn, block.ln1(x), cache, layer_idx, start)
    x = x + block.ffn(block.ln2(x))
    return x

@torch.no_grad()
def forward_with_cache(model, ids: torch.Tensor, cache: KVcache, last_only: bool = False, hiddens: list| None=None) -> torch.Tensor:
    """TransformerLM.forward의 캐시 버전
    
    ids: (B, I) 이번에 새로 넣는 토큰
    last_only: True면 마지막 위치만 lm_head에 통과
    hiddens: 리스트를 넘기면 층마다 출력을 담아줌
    반환: logits (B, I, V) 또는 last_only면 (B, 1, V)"""
    B, I = ids.shape
    start = cache.length
    if start + I > cache.max_len:
        raise ValueError(f"캐시 칸 부족: {start} + {I} > {cache.max_len}")

    x = model.token_embeddings(ids)
    for i, block in enumerate(model.layers):
        x = block_with_cache(block, x, cache, i, start)
        if hiddens is not None:
            hiddens.append(x)

    # 길이 포인터는 모든 층을 지난 뒤 한 번만 늘ㄹ미
    cache.length = start + I

    if last_only:
        x = x[:, -1:]
    x = model.ln_final(x)
    return model.lm_head(x)

@torch.no_grad()
def generate_with_cache(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float=1.0,
    top_p: float=1.0,
    eos_id: int | None=None,
    generator: torch.Generator | None=None,
    max_len: int | None=None,
) -> list[int]:
    """decoding.generate의 캐시 버전. 프롬프트 뒤에 새로 만든 토큰 ID 리스트를 돌려줌(eos 넣지 않음)"""

    model.eval()
    device = next(model.parameters()).device
    max_len = max_len or model.context_length
    if len(prompt_ids) >= max_len:
        raise ValueError(f"프롬프트 {len(prompt_ids)}토큰이 max_len {max_len} 이상")
    max_new_tokens = min(max_new_tokens, max_len - len(prompt_ids))

    cache = KVcache.for_model(model, batch=1, max_len=max_len)

    # prefill: 프롬프트 전체를 한번에. 마지막 위치 logits로 첫 토큰을 뽑음
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)   # (1, p)
    logits = forward_with_cache(model, ids, cache, last_only=True)      # (1, 1, V)

    new_ids : list[int] = []
    for step in range(max_new_tokens):
        next_id = sample_next_token(logits[0, -1], temperature, top_p, generator)
        tok = int(next_id)
        if eos_id is not None and tok == eos_id:
            break
        new_ids.append(tok)
        if step == max_new_tokens - 1:
            break
        # decode: 방금 뽑은 토큰하나만 넣음
        logits = forward_with_cache(model, next_id.view(1, 1), cache)
    return new_ids