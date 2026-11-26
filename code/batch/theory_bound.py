"""11/26 이론값 계산과 실측 대비 (원래 14:45, 16:30 블록)

위치: assignment1-basics/scratch/theory_bound.py
실행 (과제 폴더 맨 위에서): uv run python scratch/theory_bound.py
옵션: --run ts_base, --bench runs/batch/batch_<시각>.json (생략하면 가장 최근 것)

하는 일
  [1] 체크포인트에서 파라미터 수를 세고, decode 한 스텝에 읽어야 하는 가중치 바이트를 정밀도별로 계산
  [2] 대역폭 하한: 토큰당(스텝당) 시간 >= 읽을 바이트 / 896 GB/s  (fp32, bf16, fp8)
  [3] 배치 곡선 실측(bench_batch 결과)과 비교:
      스텝마다 읽는 바이트 = 가중치 + 배치 B개 줄의 KV 캐시
      달성 대역폭 = 읽은 바이트 / 실측 스텝 시간,  이론 대비 % = 달성 / 896
      연산 하한도 같이 계산해서, 어느 B부터 연산이 더 큰 하한이 되는지

주의: 아래 GPU 사양 숫자는 제조사 공개 사양과 프로젝트 배경정보 기준이며, 직접 측정한 값이 아니다.
"""
import argparse
import json
from pathlib import Path

import torch

BW = 896e9                       # 메모리 대역폭 바이트/초 (사양, 측정 아님)
PEAK = {                         # 행렬 곱 최대 연산 FLOP/초 (사양 기준, 측정 아님)
    "fp32": 44e12,               # 텐서 코어를 안 쓰는 일반 fp32 (TF32 끔) 약 44 TFLOPS
    "bf16": 88e12,               # 배경정보 기준
    "fp8": 352e12,               # 배경정보 기준
}
NBYTES = {"fp32": 4, "bf16": 2, "fp8": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ts_base")
    ap.add_argument("--bench", default=None)
    args = ap.parse_args()

    run_dir = Path("runs") / args.run
    cfg = json.loads((run_dir / "config.json").read_text())
    sd = torch.load(run_dir / "ckpt_final.pt", map_location="cpu")["model"]

    # ---- [1] 파라미터 수와 한 스텝에 읽는 가중치 ----
    total = sum(t.numel() for t in sd.values())
    emb = sum(t.numel() for k, t in sd.items() if "token_embeddings" in k)
    d, L, V = cfg["d_model"], cfg["num_layers"], cfg["vocab_size"]
    # 임베딩 표는 토큰마다 한 줄(d_model개)만 읽는다. 나머지(층들, ln_final, lm_head)는 스텝마다 전부 읽는다
    read_params = total - emb + d
    flops_tok = 2 * (total - emb)                      # 곱셈 + 덧셈 = 파라미터당 2 FLOP (attention 점수 계산 제외)
    kv_tok = 2 * L * d                                 # 토큰 하나가 캐시에 남기는 숫자 수 (K와 V, 층마다 d_model)
    print(f"모델 {args.run}: 층 {L}, d_model {d}, V {V}")
    print(f"[1] 파라미터 {total / 1e6:.2f}M (임베딩 {emb / 1e6:.2f}M), "
          f"한 스텝에 읽는 가중치 {read_params / 1e6:.2f}M개, 토큰당 연산 {flops_tok / 1e6:.1f} MFLOP, "
          f"KV 캐시 토큰당 {kv_tok}개 숫자\n")

    # ---- [2] 정밀도별 하한 (배치 1, 캐시 읽기는 작아서 뺌) ----
    print("[2] 배치 1의 토큰당 시간 하한 (가중치만 읽는다고 할 때)")
    print(f"{'정밀도':>6} | {'가중치 MB':>9} | {'대역폭 하한 us':>13} | {'연산 하한 us':>11} | 더 큰 쪽")
    for p, nb in NBYTES.items():
        w = read_params * nb
        t_mem, t_cmp = w / BW, flops_tok / PEAK[p]
        print(f"{p:>6} | {w / 1e6:9.1f} | {t_mem * 1e6:13.1f} | {t_cmp * 1e6:11.2f} | "
              f"{'대역폭' if t_mem > t_cmp else '연산'}")
    # 연산이 대역폭을 넘어서는 배치 크기 (가중치만 고려): B * flops / PEAK = w / BW
    print("    연산 하한이 대역폭 하한을 넘는 배치 크기 (가중치만 고려): "
          + ", ".join(f"{p} B ≈ {read_params * nb / BW / (flops_tok / PEAK[p]):.0f}"
                      for p, nb in NBYTES.items()))

    # ---- [3] 배치 곡선 실측과 비교 ----
    if args.bench:
        bench_path = Path(args.bench)
    else:
        cands = sorted(Path("runs/batch").glob("*.json"))
        if not cands:
            print("\nruns/batch/ 에 결과 JSON이 없어 [3]을 건너뜀")
            return
        bench_path = cands[-1]
    bench = json.loads(bench_path.read_text())
    dt = bench["config"]["dtype"]
    nb = NBYTES[dt]
    c = bench["config"]["curve"]
    ctx_avg = c["prompt_len"] + c["decode_steps"] / 2  # decode 동안 캐시 길이의 평균 (64 -> 128이면 96)
    print(f"\n[3] 실측 대비 ({bench_path}, {dt}, 프롬프트 {c['prompt_len']}, decode {c['decode_steps']}스텝, "
          f"평균 캐시 길이 {ctx_avg:.0f})")
    head = (f"{'B':>4} | {'실측 ms':>7} | {'읽는 MB':>8} (KV {'%':>3}) | {'대역폭 하한 ms':>13} | "
            f"{'연산 하한 ms':>11} | {'달성 GB/s':>9} | {'이론 대비':>8} | {'실측 tok/s':>10} | {'이론 최대 tok/s':>14}")
    print(head)
    print("-" * len(head))
    rows = []
    for r in bench["curve"]:
        B = r["B"]
        w = read_params * nb
        kv = B * ctx_avg * kv_tok * nb                 # B개 줄이 각자 자기 캐시를 읽음
        byts = w + kv
        attn_flops = B * 4 * ctx_avg * d * L           # q K^T 와 확률 x V, 층마다
        t_mem = byts / BW
        t_cmp = (B * flops_tok + attn_flops) / PEAK[dt]
        t_meas = r["step_ms"] / 1e3
        t_min = max(t_mem, t_cmp)
        ach = byts / t_meas
        row = dict(B=B, step_ms=r["step_ms"], bytes_mb=byts / 1e6, kv_pct=100 * kv / byts,
                   mem_bound_ms=t_mem * 1e3, compute_bound_ms=t_cmp * 1e3, achieved_gbs=ach / 1e9,
                   bw_pct=100 * ach / BW, tok_s=B / t_meas, max_tok_s=B / t_min)
        rows.append(row)
        print(f"{B:>4} | {r['step_ms']:7.2f} | {byts / 1e6:8.1f} ({row['kv_pct']:3.0f}) | {t_mem * 1e3:13.3f} | "
              f"{t_cmp * 1e3:11.3f} | {ach / 1e9:9.1f} | {row['bw_pct']:7.1f}% | {row['tok_s']:10.0f} | "
              f"{row['max_tok_s']:14.0f}")

    out = bench_path.with_name(bench_path.stem + "_theory.json")
    out.write_text(json.dumps(dict(run=args.run, bench=str(bench_path), bw=BW, peak=PEAK,
                                   read_params=read_params, flops_tok=flops_tok, kv_tok=kv_tok,
                                   ctx_avg=ctx_avg, rows=rows), ensure_ascii=False, indent=2))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
