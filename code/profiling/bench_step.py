"""내 모델 학습 한 스텝 벤치마크: forward, backward, optimizer 시간과 GPU 메모리 (손계산과 대조)

위치: ~/llm-study/assignment1-basics/scratch/bench_step.py
실행 (과제 폴더 맨 위에서):
    uv run python scratch/bench_step.py                       # ts_base 설정 (runs/ts_base/config.json)
    uv run python scratch/bench_step.py --run ts_L8           # 다른 모델 설정
    uv run python scratch/bench_step.py --precision fp32      # 정밀도 바꾸기
    uv run python scratch/bench_step.py --batch_size 32       # 배치 크기 바꾸기
결과: 화면 출력 + runs/bench/<run>_<precision>_b<batch>.json

- 가중치는 무작위(학습된 체크포인트를 불러오지 않음). 속도와 메모리는 가중치 값과 상관없음
- 입력도 무작위 토큰. 데이터 읽는 시간은 빼고 GPU 계산만 잼
"""
import argparse
import json
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GIB = 1024**3
BF16_SPEC_TFLOPS = 88.0   # 배경정보의 스펙 숫자 (측정 전 숫자)


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def mem():
    return torch.cuda.memory_allocated() if DEVICE == "cuda" else 0


def load_config(run: str) -> dict:
    """학습 때 저장한 config.json을 읽음. 없으면 핸드아웃 7.2.1 기본값"""
    path = Path("runs") / run / "config.json"
    default = dict(vocab_size=10000, context_length=256, d_model=512, num_layers=4,
                   num_heads=16, d_ff=1344, rope_theta=10000.0, batch_size=64)
    if path.exists():
        cfg = json.loads(path.read_text())
        print(f"설정: {path}")
    else:
        cfg = {}
        print(f"설정 파일 {path}가 없어서 핸드아웃 기본값 사용")
    return {k: cfg.get(k, v) for k, v in default.items()}


def hand_estimate(c: dict, n_params: int) -> dict:
    """11/13 adamw_accounting 식 (모든 텐서 fp32 가정, 원소 개수 x 4바이트)
    활성값 = L x [8BTd + 4BTF + 2BhT^2] + BTd + 2BTV  (F는 실제 d_ff)"""
    B, T, d = c["batch_size"], c["context_length"], c["d_model"]
    L, h, V, F = c["num_layers"], c["num_heads"], c["vocab_size"], c["d_ff"]
    per_layer = 8 * B * T * d + 4 * B * T * F + 2 * B * h * T * T
    act = L * per_layer + B * T * d + 2 * B * T * V
    return {
        "static_16P": 16 * n_params,                 # 파라미터 + 기울기 + m + v
        "act_per_layer": 4 * per_layer,
        "act_total": 4 * act,
        "peak": 16 * n_params + 4 * act,
    }


def train_flops(c: dict, n_params: int) -> float:
    """한 스텝 학습 연산량 근사: 6 x (입력 임베딩 뺀 파라미터) x 토큰 수 + attention 행렬 곱
    6 = forward 2 + backward 4. attention의 QK^T와 (확률 x V)는 파라미터가 없어서 따로 더함"""
    B, T, d, L, V = c["batch_size"], c["context_length"], c["d_model"], c["num_layers"], c["vocab_size"]
    n_matmul = n_params - V * d                      # 입력 임베딩은 표에서 꺼내기만 하고 곱하지 않음
    attn = L * 4 * B * T * T * d                     # forward: QK^T 2BT^2d + 확률xV 2BT^2d
    return 6 * n_matmul * B * T + 3 * attn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="ts_base")
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--batch_size", type=int, default=None, help="안 주면 config.json 값")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--trials", type=int, default=20)
    args = p.parse_args()

    c = load_config(args.run)
    if args.batch_size:
        c["batch_size"] = args.batch_size
    B, T, V = c["batch_size"], c["context_length"], c["vocab_size"]

    if DEVICE == "cuda":
        print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}, CUDA {torch.version.cuda}")
        torch.cuda.reset_peak_memory_stats()
    print(f"모델 {args.run}: L={c['num_layers']}, d={c['d_model']}, h={c['num_heads']}, d_ff={c['d_ff']}, "
          f"V={V} | 배치 {B} x 길이 {T} | {args.precision} | 워밍업 {args.warmup}, 측정 {args.trials}")

    # ---- 모델, 옵티마이저, 무작위 입력 ----
    m_start = mem()
    model = TransformerLM(c["vocab_size"], c["context_length"], c["d_model"], c["num_layers"],
                          c["num_heads"], c["d_ff"], c["rope_theta"], device=DEVICE)
    n_params = sum(q.numel() for q in model.parameters())
    m_model = mem()
    opt = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    x = torch.randint(0, V, (B, T), device=DEVICE)
    y = torch.randint(0, V, (B, T), device=DEVICE)

    if args.precision == "bf16":
        autocast = lambda: torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)
    else:
        autocast = nullcontext

    # ---- 한 스텝을 세 조각으로 ----
    state = {}

    def forward():
        with autocast():
            logits = model(x)
        state["loss"] = cross_entropy(logits.float(), y)   # train.py와 같이 loss는 fp32로

    def backward():
        state["loss"].backward()

    def optimizer():
        gradient_clipping(model.parameters(), 1.0)
        opt.step()

    def timed(fn) -> float:
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        return (time.perf_counter() - t0) * 1000

    # ---- 1) 조각별 시간: 조각마다 synchronize ----
    parts = {"forward": [], "backward": [], "optimizer": []}
    for i in range(args.warmup + args.trials):
        opt.zero_grad(set_to_none=True)
        tf, tb, to = timed(forward), timed(backward), timed(optimizer)
        if i >= args.warmup:
            parts["forward"].append(tf)
            parts["backward"].append(tb)
            parts["optimizer"].append(to)
    m_after_step = mem()   # 파라미터 + 기울기(아직 남아 있음) + m, v

    # ---- 2) 스텝 전체 시간: 스텝 끝에서만 synchronize (train.py에 가까운 방식) ----
    def full_step():
        opt.zero_grad(set_to_none=True)
        forward()
        backward()
        optimizer()

    whole = []
    for i in range(args.warmup + args.trials):
        t = timed(full_step)
        if i >= args.warmup:
            whole.append(t)

    # ---- 3) 최대 메모리: 한 스텝만 따로 ----
    if DEVICE == "cuda":
        opt.zero_grad(set_to_none=True)
        sync()
        torch.cuda.reset_peak_memory_stats()
        full_step()
        sync()
        m_peak = torch.cuda.max_memory_allocated()
    else:
        m_peak = 0

    # ---- 출력 ----
    def ms(v):
        return statistics.mean(v), statistics.stdev(v)

    print("\n[시간] 단위 ms, 평균 (표준편차)")
    total_parts = 0.0
    for k, v in parts.items():
        mu, sd = ms(v)
        total_parts += mu
        print(f"  {k:>10}: {mu:8.2f} ({sd:.2f})")
    print(f"  {'세 조각 합':>10}: {total_parts:8.2f}")
    mu_w, sd_w = ms(whole)
    print(f"  {'스텝 전체':>10}: {mu_w:8.2f} ({sd_w:.2f})   <- 조각마다 기다리지 않은 값")
    print(f"  backward / forward = {ms(parts['backward'])[0] / ms(parts['forward'])[0]:.2f} (이론적으로 약 2)")

    tok_s = B * T / (mu_w / 1000)
    flops = train_flops(c, n_params)
    tflops = flops / (mu_w / 1000) / 1e12
    print(f"\n[처리량] {tok_s:,.0f} tok/s | 스텝당 약 {flops / 1e12:.2f} TFLOP -> {tflops:.1f} TFLOPS"
          f" | MFU (BF16 스펙 {BF16_SPEC_TFLOPS:.0f} 대비) {tflops / BF16_SPEC_TFLOPS * 100:.1f}%")

    est = hand_estimate(c, n_params)
    print(f"\n[메모리] 파라미터 {n_params:,}개")
    if DEVICE == "cuda":
        print(f"  모델만 올렸을 때      실측 {(m_model - m_start) / GIB:6.3f} GiB | 손계산 4P   {4 * n_params / GIB:6.3f} GiB")
        print(f"  스텝 뒤 (P + grad + m + v) 실측 {(m_after_step - m_start) / GIB:6.3f} GiB | 손계산 16P  {est['static_16P'] / GIB:6.3f} GiB")
        print(f"  스텝 중 최대          실측 {(m_peak - m_start) / GIB:6.3f} GiB | 손계산 합계 {est['peak'] / GIB:6.3f} GiB (fp32 가정)")
        print(f"  최대 - 16P = 활성값   실측 {(m_peak - m_start - est['static_16P']) / GIB:6.3f} GiB | 손계산 {est['act_total'] / GIB:6.3f} GiB"
              f" (층당 {est['act_per_layer'] / GIB:.3f} GiB)")
        print("  참고: 11/16 학습 summary.json의 최대 메모리 ts_L2 4.14, ts_base 5.92, ts_L8 9.49 GiB")

    out = {
        "run": args.run, "precision": args.precision, "config": c, "params": n_params,
        "device": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu",
        "torch": torch.__version__, "warmup": args.warmup, "trials": args.trials,
        "ms": {k: dict(zip(("mean", "std"), ms(v))) for k, v in parts.items()},
        "ms_step": {"mean": mu_w, "std": sd_w},
        "tok_per_s": tok_s, "tflops": tflops,
        "mem_GiB": {"model": (m_model - m_start) / GIB, "after_step": (m_after_step - m_start) / GIB,
                    "peak": (m_peak - m_start) / GIB},
        "hand_GiB": {k: v / GIB for k, v in est.items()},
    }
    out_dir = Path("runs/bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.run}_{args.precision}_b{B}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
