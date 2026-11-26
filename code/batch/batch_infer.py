"""배치 추론 (11/26, 원래 11/25 16:30 + 11/26 10:00 블록)

위치: cs336_basics/batch_infer.py
앞 단계: cs336_basics/kv_cache.py (요청 하나짜리 캐시 생성, 11/24). 이 파일은 그 파일을 고치지 않는다.

구조
  SlotKVCache   : 슬롯(자리) S개짜리 캐시. 층마다 K, V를 (S, h, max_len, d_head)로 미리 잡고,
                  슬롯마다 길이를 따로 기억한다 (요청마다 프롬프트 길이가 다르니까)
  prefill       : 길이가 다른 프롬프트 여러 개를 오른쪽 패딩으로 한 줄로 맞춰 한 번에 계산.
                  패딩 칸은 마스크로 가린다
  decode_step   : 모든 슬롯이 토큰 1개씩 동시에. 슬롯마다 위치(= 자기 길이)가 다르다
  run_static    : 정적 배치. 요청을 B개씩 묶어 묶음 안의 가장 긴 요청이 끝날 때까지 돌리고 다음 묶음
  run_continuous: continuous batching. 매 스텝 끝난 요청을 빼고 빈 슬롯에 대기 요청을 바로 넣는다

이번 버전은 greedy만 지원한다 (정확도 비교와 속도 측정용).
"""
import time
from collections import deque

import torch
from einops import rearrange

from cs336_basics.model import scaled_dot_product_attention


class SlotKVCache:
    """슬롯 S개짜리 KV 캐시. 슬롯 하나가 요청 하나를 맡는다"""

    def __init__(self, model, num_slots: int, max_len: int | None = None):
        attn = model.layers[0].attn
        p = next(model.parameters())
        self.num_slots = num_slots
        self.max_len = max_len or model.context_length
        self.device = p.device
        shape = (num_slots, attn.num_heads, self.max_len, attn.d_head)
        self.k = [torch.zeros(shape, device=p.device, dtype=p.dtype) for _ in model.layers]
        self.v = [torch.zeros(shape, device=p.device, dtype=p.dtype) for _ in model.layers]
        # 슬롯마다 채운 칸 수. GPU 쪽(계산에 씀)과 CPU 쪽(반복문 판단에 씀, GPU를 기다리지 않으려고) 두 벌
        self.lens_t = torch.zeros(num_slots, dtype=torch.long, device=p.device)
        self.lens = [0] * num_slots

    def free(self, s: int):
        """슬롯 s를 비운다. 칸의 값은 지우지 않고 길이만 0으로 (다음 요청이 덮어쓴다)"""
        self.lens[s] = 0
        self.lens_t[s] = 0

    def reset(self):
        self.lens = [0] * self.num_slots
        self.lens_t.zero_()

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


def _qkv(attn, x):
    """x (S, I, d_model) -> q, k, v 각각 (S, h, I, d_head)"""
    q = rearrange(attn.q_proj(x), "... seq (h d) -> ... h seq d", h=attn.num_heads)
    k = rearrange(attn.k_proj(x), "... seq (h d) -> ... h seq d", h=attn.num_heads)
    v = rearrange(attn.v_proj(x), "... seq (h d) -> ... h seq d", h=attn.num_heads)
    return q, k, v


# 줄(슬롯)마다 다른 위치로 RoPE를 거는 방법. 처음 한 번 시험해서 정한다
#   "vec" : rope(x, 위치 (S, 1, I)) 한 번에 (내 RoPE가 브로드캐스팅을 지원할 때. 빠름)
#   "loop": 줄마다 따로 rope를 부르고 이어 붙임 (항상 되지만 줄 수만큼 작업이 늘어남)
ROPE_MODE = None


def rope_rows(rope, x, pos):
    """x (S, h, I, d), pos (S, I): 줄 s는 pos[s] 위치로 회전"""
    global ROPE_MODE
    if ROPE_MODE == "vec":
        return rope(x, pos.unsqueeze(1))
    loop = torch.cat([rope(x[s:s + 1], pos[s]) for s in range(x.shape[0])])
    if ROPE_MODE == "loop":
        return loop
    # 처음 한 번: 한 번에 하는 방식이 줄마다 하는 방식과 같은 값을 내는지 확인
    try:
        vec = rope(x, pos.unsqueeze(1))
        same = vec.shape == loop.shape and torch.allclose(vec, loop, rtol=1e-5, atol=1e-6)
    except Exception:
        same = False
    ROPE_MODE = "vec" if same else "loop"
    print(f"[RoPE] 줄마다 다른 위치: {'한 번에 계산 (vec)' if same else '줄마다 따로 계산 (loop)'}")
    return loop


@torch.no_grad()
def prefill(model, cache: SlotKVCache, slots: list[int], prompts: list[list[int]]) -> torch.Tensor:
    """길이가 다른 프롬프트들을 한 번에 prefill 해서 slots에 캐시를 채운다

    오른쪽 패딩: 모든 줄을 위치 0부터 시작하게 두고, 짧은 줄의 뒤를 아무 토큰(0)으로 채운다.
      줄 0: [A B C D]          길이 4
      줄 1: [E F _ _]          길이 2, _는 패딩
    모든 줄의 위치가 0, 1, 2, ...로 같아서 RoPE는 한 번에 걸 수 있다.
    마스크 = causal(앞만 보기) AND 패딩 key 가리기.
    반환: 줄마다 자기 진짜 마지막 토큰 위치의 logits (n, V)
    """
    n = len(prompts)
    lens = [len(p) for p in prompts]
    P = max(lens)
    if P > cache.max_len:
        raise ValueError(f"프롬프트 {P}토큰이 max_len {cache.max_len}보다 김")
    dev = cache.device

    ids_cpu = torch.zeros((n, P), dtype=torch.long)       # 0 = 패딩 자리 (마스크로 가려져 값은 상관없음)
    for i, p in enumerate(prompts):
        ids_cpu[i, :len(p)] = torch.tensor(p, dtype=torch.long)
    ids = ids_cpu.to(dev)                                  # GPU로 한 번에 옮김
    lens_t = torch.tensor(lens, dtype=torch.long, device=dev)
    slot_t = torch.tensor(slots, dtype=torch.long, device=dev)

    pos = torch.arange(P, device=dev)
    causal = pos[None, :] <= pos[:, None]                  # (P, P)  key j <= query i
    key_ok = pos[None, :] < lens_t[:, None]                # (n, P)  key j가 진짜 토큰인가
    mask = causal[None, None] & key_ok[:, None, None, :]   # (n, 1, P, P), 헤드 축은 브로드캐스팅
    # 패딩 query(줄 1의 위치 2, 3)도 앞쪽 진짜 key는 볼 수 있어서 행 전체가 가려지는 일은 없다 (NaN 방지)

    x = model.token_embeddings(ids)                        # (n, P, d_model)
    for l, block in enumerate(model.layers):
        attn = block.attn
        q, k, v = _qkv(attn, block.ln1(x))                 # (n, h, P, d)
        if attn.rope is not None:
            q = attn.rope(q, pos)
            k = attn.rope(k, pos)
        # 슬롯 번호로 골라서 0 ~ P-1 칸에 쓴다. 짧은 줄의 패딩 칸에도 쓰레기 값이 들어가지만
        # decode에서 그 칸을 덮어쓰거나 마스크로 가린다
        cache.k[l][slot_t, :, :P] = k
        cache.v[l][slot_t, :, :P] = v
        out = scaled_dot_product_attention(q, k, v, mask)
        x = x + attn.output_proj(rearrange(out, "... h seq d -> ... seq (h d)"))
        x = x + block.ffn(block.ln2(x))

    last = x[torch.arange(n, device=dev), lens_t - 1]      # (n, d_model) 줄마다 진짜 마지막 토큰
    logits = model.lm_head(model.ln_final(last))           # (n, V)

    cache.lens_t[slot_t] = lens_t
    for s, L in zip(slots, lens):
        cache.lens[s] = L
    return logits


@torch.no_grad()
def decode_step(model, cache: SlotKVCache, tokens: torch.Tensor, active: list[bool]) -> torch.Tensor:
    """모든 슬롯이 토큰 1개씩 동시에 decode

    tokens: (S,) 슬롯마다 방금 뽑은 토큰
    active: 슬롯마다 아직 생성 중인가. 끝난 슬롯이나 빈 슬롯도 계산은 같이 하고(배치 모양을 고정),
            길이만 늘리지 않는다. 그 계산이 정적 배치의 낭비다
    반환: (S, V)
    """
    S = cache.num_slots
    dev = cache.device
    pos = cache.lens_t.clone()                             # (S,) 새 토큰 위치 = 슬롯마다 지금 길이
    L = max(cache.lens) + 1                                # 이번에 읽을 칸 수 (가장 긴 슬롯 기준)
    if L > cache.max_len:
        raise ValueError(f"캐시 칸 부족: {L} > {cache.max_len}")
    rows = torch.arange(S, device=dev)
    key_ok = torch.arange(L, device=dev)[None, :] <= pos[:, None]   # (S, L) 자기 길이까지만
    mask = key_ok[:, None, None, :]                        # (S, 1, 1, L)

    x = model.token_embeddings(tokens.view(S, 1))          # (S, 1, d_model)
    for l, block in enumerate(model.layers):
        attn = block.attn
        q, k, v = _qkv(attn, block.ln1(x))                 # (S, h, 1, d)
        if attn.rope is not None:
            q = rope_rows(attn.rope, q, pos.view(S, 1))
            k = rope_rows(attn.rope, k, pos.view(S, 1))
        # 슬롯 s의 pos[s]번 칸에 쓴다 (슬롯마다 다른 칸)
        cache.k[l][rows, :, pos] = k[:, :, 0]
        cache.v[l][rows, :, pos] = v[:, :, 0]
        K = cache.k[l][:, :, :L]                           # (S, h, L, d)
        V = cache.v[l][:, :, :L]
        out = scaled_dot_product_attention(q, K, V, mask)
        x = x + attn.output_proj(rearrange(out, "... h seq d -> ... seq (h d)"))
        x = x + block.ffn(block.ln2(x))

    logits = model.lm_head(model.ln_final(x[:, 0]))        # (S, V)
    cache.lens_t += torch.tensor(active, dtype=torch.long, device=dev)
    for s in range(S):
        if active[s]:
            cache.lens[s] += 1
    return logits


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def run_static(model, prompts: list[list[int]], max_new: list[int], batch_size: int):
    """정적 배치: B개씩 묶어서, 묶음의 모든 요청이 끝나야 다음 묶음으로"""
    N = len(prompts)
    cache = SlotKVCache(model, batch_size)
    dev = cache.device
    outs = [[] for _ in range(N)]
    t_first, t_done = [None] * N, [None] * N
    steps = row_steps = useful = 0

    _sync(dev)
    t0 = time.perf_counter()
    for g in range(0, N, batch_size):
        reqs = list(range(g, min(g + batch_size, N)))
        cache.reset()
        tokens = torch.zeros(batch_size, dtype=torch.long, device=dev)
        tokens[:len(reqs)] = torch.argmax(prefill(model, cache, list(range(len(reqs))),
                                                  [prompts[r] for r in reqs]), dim=-1)
        active = [s < len(reqs) for s in range(batch_size)]
        while True:
            tok = tokens.tolist()                          # 여기서 GPU를 기다림 (스텝마다 한 번)
            now = time.perf_counter() - t0
            for s, r in enumerate(reqs):
                if not active[s]:
                    continue
                outs[r].append(tok[s])
                if t_first[r] is None:
                    t_first[r] = now
                if len(outs[r]) >= max_new[r]:
                    active[s] = False
                    t_done[r] = now
            if not any(active):
                break
            tokens = torch.argmax(decode_step(model, cache, tokens, active), dim=-1)
            steps += 1
            row_steps += batch_size
            useful += sum(active)
    total = time.perf_counter() - t0
    return outs, dict(total_s=total, t_first=t_first, t_done=t_done,
                      decode_steps=steps, row_steps=row_steps, useful_row_steps=useful)


@torch.no_grad()
def run_continuous(model, prompts: list[list[int]], max_new: list[int], num_slots: int):
    """continuous batching: 매 스텝 끝난 요청을 빼고, 빈 슬롯에 대기 요청을 넣는다

    한 스텝의 순서
      1) 빈 슬롯이 있고 대기 요청이 있으면, 들어갈 요청들을 한 번에 prefill (첫 토큰이 나옴)
      2) 모든 슬롯의 이번 토큰을 CPU로 가져와 요청별 출력에 붙이고, 다 만든 요청은 슬롯을 비움
      3) 남은 슬롯 전부 decode 한 스텝
    1)이 도는 동안 다른 슬롯의 decode는 기다린다. 이 문제를 푸는 게 12/5 Sarathi-Serve(chunked prefill)
    """
    N = len(prompts)
    cache = SlotKVCache(model, num_slots)
    dev = cache.device
    outs = [[] for _ in range(N)]
    t_first, t_done = [None] * N, [None] * N
    queue = deque(range(N))                                # 대기 줄 (앞에서부터 꺼냄)
    slot_req = [None] * num_slots                          # 슬롯마다 맡은 요청 번호 (None = 빈 슬롯)
    tokens = torch.zeros(num_slots, dtype=torch.long, device=dev)
    steps = row_steps = useful = prefills = 0

    _sync(dev)
    t0 = time.perf_counter()
    while True:
        # 1) 빈 슬롯 채우기
        free = [s for s in range(num_slots) if slot_req[s] is None]
        if free and queue:
            take = free[:len(queue)]
            reqs = [queue.popleft() for _ in take]
            for s, r in zip(take, reqs):
                slot_req[s] = r
            logits = prefill(model, cache, take, [prompts[r] for r in reqs])
            tokens[torch.tensor(take, device=dev)] = torch.argmax(logits, dim=-1)
            prefills += 1

        # 2) 이번 토큰 받기, 끝난 요청 빼기
        tok = tokens.tolist()
        now = time.perf_counter() - t0
        for s in range(num_slots):
            r = slot_req[s]
            if r is None:
                continue
            outs[r].append(tok[s])
            if t_first[r] is None:
                t_first[r] = now
            if len(outs[r]) >= max_new[r]:
                t_done[r] = now
                slot_req[s] = None
                cache.free(s)

        active = [r is not None for r in slot_req]
        if not any(active):
            if not queue:
                break
            continue                                       # 전부 끝났는데 대기 요청이 있으면 바로 채우러

        # 3) decode 한 스텝
        tokens = torch.argmax(decode_step(model, cache, tokens, active), dim=-1)
        steps += 1
        row_steps += num_slots
        useful += sum(active)
    total = time.perf_counter() - t0
    return outs, dict(total_s=total, t_first=t_first, t_done=t_done, decode_steps=steps,
                      row_steps=row_steps, useful_row_steps=useful, prefill_calls=prefills)
