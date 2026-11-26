"""11/26 배치 추론 정확도 검증

위치: assignment1-basics/scratch/check_batch.py
실행 (과제 폴더 맨 위에서): uv run python scratch/check_batch.py
옵션: --run ts_L8, --batch 4

확인하는 것 (fp32, TF32 끔, greedy)
  [1] 줄마다 다른 위치로 RoPE를 거는 방식 (한 번에 되는지, 줄마다 따로인지)
  [2] 정적 배치 결과가 요청 하나씩 생성한 결과(11/24 generate_with_cache)와 같은지
  [3] continuous batching 결과도 같은지
  [4] decode 스텝 수와 낭비: 슬롯 x 스텝 중 실제로 토큰을 만든 비율
  [5] 걸린 시간 미리 보기 (정식 측정은 다음 블록)

요청 12개는 프롬프트 길이(3~128)와 생성 길이(10~200)를 일부러 제각각으로 골랐다.
패딩과 마스크가 틀리면 짧은 프롬프트가 긴 프롬프트의 패딩을 보게 되어 결과가 달라진다.
"""
import argparse
import json
import time
from pathlib import Path

import torch

import cs336_basics.batch_infer as bi
from cs336_basics.kv_cache import generate_with_cache
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

EOT = "<|endoftext|>"
STORY = ("Once upon a time, there was a little girl named Lily. She loved to play in the park with her dog, Max. "
         "One day, Lily saw a big red ball under a tree. She ran to the ball and kicked it high into the sky. "
         "Max barked and jumped. The ball fell into the pond, and Lily was sad. Then a kind old man came by "
         "with a long stick. He helped Lily get the ball back. Lily said thank you and gave him a big hug. ")

# (프롬프트 길이 P, 생성 길이 n). P + n <= 256
REQS = [(4, 30), (9, 120), (17, 10), (33, 64), (60, 100), (100, 20),
        (5, 150), (40, 50), (128, 80), (12, 128), (75, 40), (3, 200)]


def load_model(run_dir: Path, device: str):
    cfg = json.loads((run_dir / "config.json").read_text())
    model = TransformerLM(
        cfg["vocab_size"], cfg["context_length"], cfg["d_model"], cfg["num_layers"],
        cfg["num_heads"], cfg["d_ff"], cfg["rope_theta"], device=device,
    )
    ckpt = torch.load(run_dir / "ckpt_final.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def ok(c) -> str:
    return "통과" if c else "실패"


def compare(name, outs, refs):
    """요청마다 토큰이 같은지. 다르면 몇 번째 토큰에서 갈라졌는지"""
    same = 0
    for i, (a, b) in enumerate(zip(outs, refs)):
        if a == b:
            same += 1
        else:
            d = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(f"     요청 {i} (P {REQS[i][0]}, n {REQS[i][1]}): {d}번째 생성 토큰에서 갈라짐 "
                  f"(길이 {len(a)} 대 {len(b)})")
    print(f"{name}: 단독 생성과 일치 {same}/{len(refs)}", ok(same == len(refs)))
    return same == len(refs)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ts_base")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(Path("runs") / args.run, args.device)
    tok = Tokenizer.from_files("out/ts_bpe/vocab.pkl", "out/ts_bpe/merges.pkl", special_tokens=[EOT])
    story = tok.encode(STORY * 4)
    # 요청마다 이야기의 다른 곳에서 잘라 내용이 겹치지 않게
    prompts = [story[i * 23: i * 23 + P] for i, (P, _) in enumerate(REQS)]
    max_new = [n for _, n in REQS]
    assert all(len(p) == P for p, (P, _) in zip(prompts, REQS)), "이야기가 짧아 프롬프트를 못 자름"
    print(f"모델 {args.run}, 배치(슬롯) {args.batch}, 요청 {len(REQS)}개, 생성 토큰 합 {sum(max_new)}\n")

    # 기준: 요청 하나씩 (11/24 코드)
    refs = [generate_with_cache(model, p, n, temperature=0) for p, n in zip(prompts, max_new)]

    st_out, st = bi.run_static(model, prompts, max_new, args.batch)
    print(f"[1] RoPE 방식: {bi.ROPE_MODE}")
    ok2 = compare("[2] 정적 배치", st_out, refs)
    ct_out, ct = bi.run_continuous(model, prompts, max_new, args.batch)
    ok3 = compare("[3] continuous batching", ct_out, refs)

    def waste(d):
        return (f"decode {d['decode_steps']}스텝, 슬롯 x 스텝 {d['row_steps']} 중 토큰을 만든 칸 "
                f"{d['useful_row_steps']} ({100 * d['useful_row_steps'] / d['row_steps']:.0f}%)")
    print(f"\n[4] 정적: {waste(st)}")
    print(f"    continuous: {waste(ct)}, prefill {ct['prefill_calls']}번")

    # [5] 시간 미리 보기: 위에서 한 번씩 돌았으니 워밍업은 된 셈. 한 번씩만 재므로 대략적인 값
    sync()
    t0 = time.perf_counter()
    for p, n in zip(prompts, max_new):
        generate_with_cache(model, p, n, temperature=0)
    sync()
    t_single = time.perf_counter() - t0
    _, st = bi.run_static(model, prompts, max_new, args.batch)
    _, ct = bi.run_continuous(model, prompts, max_new, args.batch)
    tot = sum(max_new)
    print(f"\n[5] 시간 (한 번씩, 대략): 하나씩 {t_single * 1e3:.0f} ms ({tot / t_single:.0f} tok/s), "
          f"정적 {st['total_s'] * 1e3:.0f} ms ({tot / st['total_s']:.0f} tok/s), "
          f"continuous {ct['total_s'] * 1e3:.0f} ms ({tot / ct['total_s']:.0f} tok/s)")

    print(f"\n정리: 정적 {ok(ok2)}, continuous {ok(ok3)}")


if __name__ == "__main__":
    main()
