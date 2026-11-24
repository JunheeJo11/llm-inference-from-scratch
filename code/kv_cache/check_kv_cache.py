"""11/24 KV 캐시 정확도 검증

위치: assignment1-basics/scratch/check_kv_cache.py
실행 (과제 폴더 맨 위에서): uv run python scratch/check_kv_cache.py
옵션: --run ts_L8 (다른 모델), --device cpu

확인하는 것 (전부 fp32)
  [1] 캐시 크기가 설계 손계산과 같은지 (ts_base 256칸이면 fp32 4 MiB)
  [2] greedy 생성 토큰이 캐시 유무로 완전히 같은지 (프롬프트 3개, 전체 길이 256까지)
  [3] 같은 문장을 넣었을 때 위치마다 logits 차이가 얼마인지 (teacher forcing)
  [4] 층마다 출력 차이 (어느 층에서 갈라지는지 추적용)
  [5] 온도 1 샘플링도 같은 시드면 같은 토큰이 나오는지
"""
import argparse
import json
from pathlib import Path

import torch

from cs336_basics.decoding import generate
from cs336_basics.kv_cache import KVcache, forward_with_cache, generate_with_cache
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

EOT = "<|endoftext|>"
PROMPTS = ["Once upon a time", "Lily and Tom went to the park.", "The little dog was very"]


def load_model(run_dir: Path, device: str):
    """generate.py와 같은 방식: config.json으로 모델을 만들고 ckpt_final.pt 가중치를 넣는다"""
    cfg = json.loads((run_dir / "config.json").read_text())
    model = TransformerLM(
        cfg["vocab_size"], cfg["context_length"], cfg["d_model"], cfg["num_layers"],
        cfg["num_heads"], cfg["d_ff"], cfg["rope_theta"], device=device,
    )
    ckpt = torch.load(run_dir / "ckpt_final.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def ok(c) -> str:
    return "통과" if c else "실패"


@torch.no_grad()
def full_hiddens(model, ids: torch.Tensor) -> list[torch.Tensor]:
    """캐시 없이 한 번에 넣었을 때 층마다 출력 (원래 부품 그대로)"""
    out = []
    x = model.token_embeddings(ids)
    for block in model.layers:
        x = block(x)  # token_positions 없이 부르면 0부터 위치를 매김
        out.append(x)
    return out


@torch.no_grad()
def cached_pass(model, prompt: list[int], rest: list[int]):
    """prefill(prompt) 뒤에 rest를 한 개씩 decode. 위치마다 logits와 층별 출력을 이어 붙여 돌려준다"""
    device = next(model.parameters()).device
    cache = KVcache.for_model(model)
    layer_outs = [[] for _ in model.layers]
    logits_all = []

    h = []
    logits_all.append(forward_with_cache(model, torch.tensor([prompt], device=device), cache, hiddens=h))
    for i, t in enumerate(h):
        layer_outs[i].append(t)
    for tok in rest:
        h = []
        logits_all.append(forward_with_cache(model, torch.tensor([[tok]], device=device), cache, hiddens=h))
        for i, t in enumerate(h):
            layer_outs[i].append(t)

    logits = torch.cat(logits_all, dim=1)                   # (1, L, V)
    layers = [torch.cat(o, dim=1) for o in layer_outs]      # 층마다 (1, L, d_model)
    return logits, layers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="ts_base")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False  # 진짜 fp32로 (TF32면 차이가 커져 판단이 흐려짐)
    model, cfg = load_model(Path("runs") / args.run, args.device)
    tok = Tokenizer.from_files("out/ts_bpe/vocab.pkl", "out/ts_bpe/merges.pkl", special_tokens=[EOT])
    T_max = cfg["context_length"]
    print(f"모델 {args.run}: 층 {cfg['num_layers']}, d_model {cfg['d_model']}, 헤드 {cfg['num_heads']}, "
          f"context {T_max}, 장치 {args.device}, dtype {next(model.parameters()).dtype}\n")

    # ---- [1] 캐시 크기 ----
    cache = KVcache.for_model(model)
    expect = 2 * cfg["num_layers"] * cfg["d_model"] * T_max * 4  # K와 V, 층, h*d_head, 칸 수, fp32 4바이트
    print(f"[1] 캐시 크기 {cache.nbytes() / 2**20:.2f} MiB, 손계산 {expect / 2**20:.2f} MiB:",
          ok(cache.nbytes() == expect))

    all_same = True
    worst_logit = 0.0
    for prompt in PROMPTS:
        ids = tok.encode(prompt)
        n_new = T_max - len(ids)  # 결정 3: 프롬프트 + 생성 = 256

        # ---- [2] greedy 토큰 비교 (eos로 멈추지 않게 eos_id 없이 끝까지) ----
        ref = generate(model, ids, n_new, T_max, temperature=0)          # 캐시 없음 (원래 코드)
        new = generate_with_cache(model, ids, n_new, temperature=0)      # 캐시 있음
        same = ref == new
        all_same &= same
        first_diff = next((i for i, (a, b) in enumerate(zip(ref, new)) if a != b), None)
        print(f"\n[2] \"{prompt}\" ({len(ids)}토큰) greedy {len(ref)} 대 {len(new)}토큰 일치:", ok(same),
              "" if same else f"-> {first_diff}번째 생성 토큰에서 갈라짐")

        # ---- [3] 같은 문장을 넣고 위치마다 logits 차이 ----
        rest = ref[:-1]  # 마지막 생성 토큰은 다시 넣을 일이 없음
        seq = torch.tensor([ids + rest], device=args.device)            # (1, L) L = 255
        with torch.no_grad():
            full = model(seq)
        cached, cached_layers = cached_pass(model, ids, rest)
        diff_pos = (full - cached).abs().amax(dim=-1)[0]                 # 위치마다 최대 차이 (L,)
        worst_logit = max(worst_logit, diff_pos.max().item())
        print(f"[3] logits 최대 차이: prefill 구간 {diff_pos[:len(ids)].max().item():.2e}, "
              f"decode 구간 {diff_pos[len(ids):].max().item():.2e} (가장 큰 위치 {int(diff_pos.argmax())})")

        # ---- [4] 층마다 출력 차이 ----
        ref_layers = full_hiddens(model, seq)
        per_layer = [(a - b).abs().max().item() for a, b in zip(ref_layers, cached_layers)]
        print("[4] 층별 최대 차이:", ", ".join(f"층{i} {d:.1e}" for i, d in enumerate(per_layer)))

    # ---- [5] 샘플링도 같은 시드면 같은가 ----
    ids = tok.encode(PROMPTS[0])
    g1, g2 = torch.Generator(args.device).manual_seed(0), torch.Generator(args.device).manual_seed(0)
    ref = generate(model, ids, 100, T_max, temperature=1.0, generator=g1)
    new = generate_with_cache(model, ids, 100, temperature=1.0, generator=g2)
    print("\n[5] 온도 1, 시드 0으로 100토큰 샘플링 일치:", ok(ref == new))
    print("    생성 예:", PROMPTS[0] + tok.decode(new[:40]), "...")

    print(f"\n정리: greedy 전부 일치 {ok(all_same)}, logits 최대 차이 {worst_logit:.2e} "
          f"({'1e-4 미만, 계산 순서 차이 수준' if worst_logit < 1e-4 else '큼, 위치와 마스크부터 확인'})")


if __name__ == "__main__":
    main()