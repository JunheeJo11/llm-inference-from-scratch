"""11/24 KV 캐시 효과 측정: 캐시 없음(decoding.generate) 대 캐시 있음(kv_cache.generate_with_cache)

위치: assignment1-basics/scratch/bench_kv_cache.py
실행 (과제 폴더 맨 위에서):
    uv run --with matplotlib python scratch/bench_kv_cache.py                 # 기본: ts_base, fp32, 길이 5단계
    uv run --with matplotlib python scratch/bench_kv_cache.py --dtype bf16    # 같은 측정을 bf16으로
    uv run python scratch/bench_kv_cache.py --profile                         # 토큰당 커널 수와 GPU 바쁜 비율

측정 조건 (11/17 기준과 맞춤): 배치 1, 프롬프트 "Once upon a time", greedy, eos로 멈추지 않고 정해진 길이까지
결과: 화면 표 + runs/bench_kv_cache_<run>_<dtype>.json (+ matplotlib 있으면 같은 이름 .png)
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from cs336_basics.decoding import generate
from cs336_basics.kv_cache import generate_with_cache
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

EOT = "<|endoftext|>"
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def load_model(run_dir: Path, device: str, dtype: torch.dtype):
    """generate.py와 같은 방식으로 불러온 뒤 원하는 정밀도로 바꾼다"""
    cfg = json.loads((run_dir / "config.json").read_text())
    model = TransformerLM(
        cfg["vocab_size"], cfg["context_length"], cfg["d_model"], cfg["num_layers"],
        cfg["num_heads"], cfg["d_ff"], cfg["rope_theta"], device=device,
    )
    ckpt = torch.load(run_dir / "ckpt_final.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(dtype)  # 가중치와 RoPE 표를 함께 바꾼다. 캐시는 for_model이 모델 dtype을 따라감
    model.eval()
    return model, cfg


def sync(device: str):
    """GPU에 걸어 둔 일이 다 끝날 때까지 기다림. 없으면 '명령 넣은 시간'만 재게 된다"""
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def measure(fn, device: str, warmup: int, repeats: int):
    """fn()을 warmup번 버리고 repeats번 재서 (초 목록, 마지막 결과)를 돌려준다"""
    out = None
    for _ in range(warmup):
        out = fn()
    secs = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        out = fn()
        sync(device)
        secs.append(time.perf_counter() - t0)
    return secs, out


def profile_run(fn, device: str):
    """fn() 한 번을 프로파일해서 (GPU 작업 수, GPU 작업 시간 합 초, 프로파일 없이 잰 벽시계 초)"""
    from torch.profiler import ProfilerActivity, profile

    fn()  # 워밍업
    sync(device)
    t0 = time.perf_counter()
    fn()
    sync(device)
    wall = time.perf_counter() - t0  # 프로파일러는 자체 비용이 있어서 벽시계는 따로 잰다

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        sync(device)
    gpu = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    gpu_sec = sum(e.time_range.elapsed_us() for e in gpu) / 1e6
    return len(gpu), gpu_sec, wall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="ts_base")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--lengths", default="16,32,64,128,252", help="생성 길이 목록, 쉼표로 구분")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    p.add_argument("--profile", action="store_true", help="표 대신 커널 수와 GPU 바쁜 비율 측정")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model, cfg = load_model(Path("runs") / args.run, args.device, DTYPES[args.dtype])
    tok = Tokenizer.from_files("out/ts_bpe/vocab.pkl", "out/ts_bpe/merges.pkl", special_tokens=[EOT])
    ids = tok.encode(args.prompt)
    P, ctx = len(ids), cfg["context_length"]
    gpu_name = torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu"
    print(f"조건: {args.run}, {args.dtype}, {gpu_name}, torch {torch.__version__}, "
          f"TF32 {torch.backends.cuda.matmul.allow_tf32}, 배치 1, 프롬프트 {P}토큰, greedy, "
          f"워밍업 {args.warmup}, 반복 {args.repeats}\n")

    # 생성 함수 두 개를 같은 모양으로 감싼다. eos_id를 안 넘겨서 항상 n개를 끝까지 만든다
    def no_cache(n):
        return lambda: generate(model, ids, n, ctx, temperature=0)

    def with_cache(n):
        return lambda: generate_with_cache(model, ids, n, temperature=0)

    if args.profile:
        n = 64
        print(f"프로파일: 생성 {n}토큰 한 번")
        print(f"{'':>8} {'GPU 작업 수':>12} {'토큰당':>8} {'GPU 시간':>10} {'벽시계':>10} {'GPU 바쁜 비율':>13}")
        for name, fn in [("캐시없음", no_cache(n)), ("캐시", with_cache(n))]:
            count, gpu_sec, wall = profile_run(fn, args.device)
            print(f"{name:>8} {count:>12,} {count / n:>8.0f} {gpu_sec * 1e3:>8.1f}ms {wall * 1e3:>8.1f}ms "
                  f"{gpu_sec / wall * 100:>12.0f}%")
        return

    lengths = [min(int(s), ctx - P) for s in args.lengths.split(",")]  # 결정 3: 전체 256 안으로
    rows = []
    head = (f"{'생성 n':>6} | {'캐시없음 ms':>14} {'tok/s':>7} | {'캐시 ms':>14} {'tok/s':>7} | "
            f"{'속도비':>6} {'입력 토큰비':>10} {'출력 일치':>8}")
    print(head)
    print("-" * len(head))
    for n in lengths:
        s0, out0 = measure(no_cache(n), args.device, args.warmup, args.repeats)
        s1, out1 = measure(with_cache(n), args.device, args.warmup, args.repeats)
        m0, sd0 = statistics.mean(s0), statistics.stdev(s0)
        m1, sd1 = statistics.mean(s1), statistics.stdev(s1)
        # 모델에 들어간 토큰 수: 캐시 없음은 매 스텝 P + i개, 캐시는 prefill P개 + decode (n - 1)개
        fed0 = n * P + n * (n - 1) // 2
        fed1 = P + n - 1
        row = dict(n=n, no_cache_ms=m0 * 1e3, no_cache_sd=sd0 * 1e3, no_cache_tps=n / m0,
                   cache_ms=m1 * 1e3, cache_sd=sd1 * 1e3, cache_tps=n / m1,
                   speedup=m0 / m1, fed_ratio=fed0 / fed1, same=out0 == out1)
        rows.append(row)
        print(f"{n:>6} | {row['no_cache_ms']:>7.1f} ±{row['no_cache_sd']:>5.1f} {row['no_cache_tps']:>7.0f} | "
              f"{row['cache_ms']:>7.1f} ±{row['cache_sd']:>5.1f} {row['cache_tps']:>7.0f} | "
              f"{row['speedup']:>5.2f}x {row['fed_ratio']:>9.1f}x {str(row['same']):>8}")

    out_base = Path("runs") / f"bench_kv_cache_{args.run}_{args.dtype}"
    meta = dict(run=args.run, dtype=args.dtype, gpu=gpu_name, torch=torch.__version__,
                tf32=torch.backends.cuda.matmul.allow_tf32, prompt=args.prompt, prompt_tokens=P,
                warmup=args.warmup, repeats=args.repeats, decoding="greedy, eos 무시")
    out_base.with_suffix(".json").write_text(json.dumps(dict(meta=meta, rows=rows), ensure_ascii=False, indent=2))
    print(f"\n저장: {out_base.with_suffix('.json')}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 없음: 그림은 건너뜀 (uv run --with matplotlib 로 실행하면 그림도 저장)")
        return
    ns = [r["n"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ns, [r["no_cache_tps"] for r in rows], "o-", label="no cache")
    axes[0].plot(ns, [r["cache_tps"] for r in rows], "o-", label="KV cache")
    axes[0].set(xlabel="generated tokens", ylabel="tokens / s", title=f"{args.run} {args.dtype} decode speed")
    axes[1].plot(ns, [r["speedup"] for r in rows], "o-", label="measured speedup")
    axes[1].plot(ns, [r["fed_ratio"] for r in rows], "o--", label="tokens fed ratio (ideal)")
    axes[1].set(xlabel="generated tokens", ylabel="x (log)", yscale="log", title="speedup vs ideal")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=120)
    print(f"저장: {out_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()