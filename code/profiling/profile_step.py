"""내 모델 학습 한 스텝 프로파일: 시간이 어느 연산(커널)에 쓰이는지, forward에서 모듈별 비중

위치: ~/llm-study/assignment1-basics/scratch/profile_step.py
실행 (과제 폴더 맨 위에서):
    uv run python scratch/profile_step.py                 # ts_base, bf16
    uv run python scratch/profile_step.py --run ts_L8
결과:
    화면: (1) 오래 걸린 연산 표 (2) 커널 종류별 시간 비중 (3) forward 모듈별 시간 비중
    파일: runs/profile/<run>_<precision>_trace.json  (https://ui.perfetto.dev 에서 열어 시간 축으로 보기)
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from bench_step import DEVICE, load_config, sync   # 같은 scratch 폴더의 bench_step.py에서 가져옴
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW


def kernel_kind(name: str) -> str:
    """커널 이름으로 대략 분류 (이름 규칙은 라이브러리 버전마다 달라서 근사)"""
    n = name.lower()
    if any(k in n for k in ("gemm", "cutlass", "cublas", "xmma", "matmul", "sm80_", "sm90_", "sm100_", "sm120_")):
        return "행렬 곱(GEMM)"
    if "softmax" in n:
        return "softmax"
    if "reduce" in n:
        return "합, 최댓값 같은 줄이기(reduce)"
    if "elementwise" in n or "vectorized" in n or "unrolled" in n:
        return "원소별 연산(elementwise)"
    if "memcpy" in n or "memset" in n or "copy" in n:
        return "복사, 초기화"
    return "기타"


class ModuleTimer:
    """forward 중 모듈마다 걸린 시간을 CUDA 이벤트로 잼 (GPU 시간 축에 찍는 시계)"""

    def __init__(self, model):
        self.records = defaultdict(list)    # 이름 -> [(시작 이벤트, 끝 이벤트), ...]
        self.handles = []
        targets = []
        for top_name, top in model.named_children():
            if isinstance(top, torch.nn.ModuleList):           # Transformer block들이 담긴 목록
                for block in top:
                    for child_name, child in block.named_children():
                        targets.append((f"block.{child_name}", child))   # 모든 층의 같은 부품끼리 합산
            else:
                targets.append((top_name, top))                # 임베딩, 최종 norm, lm_head 등
        for name, mod in targets:
            self.handles.append(mod.register_forward_pre_hook(self._start(name)))
            self.handles.append(mod.register_forward_hook(self._end(name)))

    def _event(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def _start(self, name):
        def hook(module, inputs):
            self.records[name].append([self._event(), None])
        return hook

    def _end(self, name):
        def hook(module, inputs, output):
            self.records[name][-1][1] = self._event()
        return hook

    def totals_ms(self):
        sync()
        return {k: sum(s.elapsed_time(e) for s, e in v) for k, v in self.records.items()}

    def remove(self):
        for h in self.handles:
            h.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="ts_base")
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--steps", type=int, default=3, help="프로파일할 스텝 수")
    args = p.parse_args()
    assert DEVICE == "cuda", "GPU가 필요해요"

    c = load_config(args.run)
    if args.batch_size:
        c["batch_size"] = args.batch_size
    B, T, V = c["batch_size"], c["context_length"], c["vocab_size"]
    model = TransformerLM(c["vocab_size"], c["context_length"], c["d_model"], c["num_layers"],
                          c["num_heads"], c["d_ff"], c["rope_theta"], device=DEVICE)
    opt = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    x = torch.randint(0, V, (B, T), device=DEVICE)
    y = torch.randint(0, V, (B, T), device=DEVICE)
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.precision == "bf16" else nullcontext

    def step():
        opt.zero_grad(set_to_none=True)
        with record_function("1_forward"):          # 표와 trace에 이 이름으로 구간이 표시됨
            with autocast():
                logits = model(x)
            loss = cross_entropy(logits.float(), y)
        with record_function("2_backward"):
            loss.backward()
        with record_function("3_optimizer"):
            gradient_clipping(model.parameters(), 1.0)
            opt.step()

    for _ in range(3):      # 워밍업
        step()
    sync()

    # ---- (1) 프로파일러 ----
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            step()
        sync()

    print(f"\n===== (1) GPU 시간이 긴 연산 상위 20개 ({args.run}, {args.precision}, {args.steps}스텝 합계) =====")
    avg = prof.key_averages()
    try:
        print(avg.table(sort_by="cuda_time_total", row_limit=20, max_name_column_width=60))
    except Exception:
        print(avg.table(sort_by="device_time_total", row_limit=20, max_name_column_width=60))

    out_dir = Path("runs/profile")
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = out_dir / f"{args.run}_{args.precision}_trace.json"
    prof.export_chrome_trace(str(trace))
    print(f"trace 저장: {trace}")

    # ---- (2) 커널 종류별 비중 ----
    print("\n===== (2) GPU 커널 종류별 시간 비중 =====")
    try:
        by_kind = defaultdict(float)
        n_kernels = 0
        for evt in prof.events():
            if evt.device_type == torch.autograd.DeviceType.CUDA:      # GPU에서 실제로 돈 커널만
                by_kind[kernel_kind(evt.name)] += evt.time_range.elapsed_us()
                n_kernels += 1
        total = sum(by_kind.values())
        print(f"커널 실행 {n_kernels}번 ({n_kernels / args.steps:.0f}번/스텝), GPU 시간 합 {total / 1000 / args.steps:.1f} ms/스텝")
        for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<28} {v / 1000 / args.steps:8.2f} ms/스텝  {v / total * 100:5.1f}%")
    except Exception as e:
        print(f"분류 실패 ({type(e).__name__}: {e}). (1)의 표로 대신 보면 됨")

    # ---- (3) forward 모듈별 비중 ----
    print("\n===== (3) forward 안에서 모듈별 시간 (CUDA 이벤트, 층 전체 합산) =====")
    timer = ModuleTimer(model)
    n = 10
    fwd_start, fwd_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for _ in range(3):
            with autocast():
                model(x)
        timer.records.clear()
        sync()
        fwd_start.record()
        for _ in range(n):
            with autocast():
                model(x)
        fwd_end.record()
    totals = timer.totals_ms()
    timer.remove()
    fwd_total = fwd_start.elapsed_time(fwd_end) / n
    covered = 0.0
    print(f"forward 한 번 평균 {fwd_total:.2f} ms ({n}회 평균, 기울기 기록 끔)")
    for k, v in sorted(totals.items(), key=lambda kv: -kv[1]):
        per = v / n
        covered += per
        print(f"  {k:<20} {per:8.2f} ms  {per / fwd_total * 100:5.1f}%")
    print(f"  {'나머지(잔차 덧셈 등)':<20} {fwd_total - covered:8.2f} ms  {(fwd_total - covered) / fwd_total * 100:5.1f}%")


if __name__ == "__main__":
    main()
