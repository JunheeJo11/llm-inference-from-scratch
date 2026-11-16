# 5.3 Problem(training_together)

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.optimizer import AdamW
from cs336_basics.serialization import load_checkpoint, save_checkpoint

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", default="data/ts_train.npy")
    p.add_argument("--val_data", default="data/ts_valid.npy")
    # 모델
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    # 옵티마이저, 스케줄
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--lr_max", type=float, default=1e-3)
    p.add_argument("--lr_min_ratio", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # 기록과 저장
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--ckpt_interval", type=int, default=5000)
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--run_name", default="ts_base")
    p.add_argument("--resume", default=None)
    # 실행 환경
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

@torch.no_grad()
def estimate_val_loss(model, val, args, autocast):
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        x, y = get_batch(val, args.batch_size, args.context_length, args.device)
        with autocast():
            logits = model(x)
        losses.append(cross_entropy(logits.float(), y).item())
    model.train()
    return sum(losses) / len(losses)

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))

    train = np.load(args.train_data, mmap_mode="r")
    val = np.load(args.val_data, mmap_mode="r")
    print(f"train {len(train):,} 토큰, val {len(val):,} 토큰, dtype {train.dtype}")

    model = TransformerLM(
        args.vocab_size, args.context_length, args.d_model, args.num_layers,
        args.num_heads, args.d_ff, args.rope_theta, device=args.device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"파라미터 {n_params:,}개 (입력 임베딩 제외 {n_params - args.vocab_size * args.d_model:,}개)")

    opt = AdamW(
        model.parameters(), lr=args.lr_max, betas=(args.beta1, args.beta2),
        eps=args.eps, weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, opt)
        print(f"{args.resume}에서 이어서 시작: step {start_step}")

    use_bf16 = args.precision == "bf16" and args.device.startswith("cuda")
    if use_bf16:
        autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast = lambda: torch.autocast(device_type="cpu", enabled=False)  # 아무것도 안 하는 자리표시
        torch.backends.cuda.matmul.allow_tf32 = True  # fp32라도 행렬 곱은 TF32로 (Ampere 이후 GPU)

    log_path = run_dir / "log.csv"
    new_file = not log_path.exists() or start_step == 0
    log_f = open(log_path, "w" if new_file else "a", newline="")
    writer = csv.writer(log_f)
    if new_file:
        writer.writerow(["step", "tokens", "lr", "train_loss", "val_loss", "grad_norm", "step_ms", "tok_per_s", "elapsed_min"])

    tokens_per_step = args.batch_size * args.context_length
    t_start = time.perf_counter()
    t_last = t_start
    model.train()
    step = start_step - 1

    for step in range(start_step, args.max_steps):
        # 1) 이번 스텝 학습률 (4.4)
        lr = get_lr_cosine_schedule(
            step, args.lr_max, args.lr_max * args.lr_min_ratio, args.warmup_steps, args.max_steps
        )
        for group in opt.param_groups:
            group["lr"] = lr

        # 2) 배치 (5.1)
        x, y = get_batch(train, args.batch_size, args.context_length, args.device)

        # 3) forward와 loss (3장, 4.1). loss는 fp32로 계산
        with autocast():
            logits = model(x)
        loss = cross_entropy(logits.float(), y)

        # 4) backward, clipping, step (4.5, 4.3)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        do_log = (step + 1) % args.log_interval == 0 or step == start_step
        if do_log:  # 기록하는 스텝에서만 자르기 전 기울기 노름을 잼 (매 스텝 재면 느려짐)
            grad_norm = torch.sqrt(sum((p.grad.detach().float() ** 2).sum()
                                       for p in model.parameters() if p.grad is not None)).item()
        gradient_clipping(model.parameters(), args.grad_clip)
        opt.step()

        # 5) 기록
        is_eval = (step + 1) % args.eval_interval == 0 or step + 1 == args.max_steps
        if do_log or is_eval:
            train_loss = loss.item()  # .item()이 GPU 작업이 끝나길 기다리므로 여기서 시간을 잼
            now = time.perf_counter()
            n_steps = 1 if step == start_step else args.log_interval
            step_ms = (now - t_last) / n_steps * 1000
            t_last = now
            val_loss = estimate_val_loss(model, val, args, autocast) if is_eval else ""
            if is_eval:
                t_last = time.perf_counter()  # 검증에 쓴 시간은 스텝 시간에서 빼기
            row = [step + 1, (step + 1) * tokens_per_step, f"{lr:.3e}", f"{train_loss:.4f}",
                   f"{val_loss:.4f}" if val_loss != "" else "", f"{grad_norm:.3f}" if do_log else "",
                   f"{step_ms:.1f}", f"{tokens_per_step / step_ms * 1000:.0f}",
                   f"{(now - t_start) / 60:.2f}"]
            writer.writerow(row)
            log_f.flush()  # 바로 파일에 쓰기 (중간에 멈춰도 기록이 남게)
            msg = f"step {step + 1:6d} | lr {lr:.2e} | train {train_loss:.4f}"
            if is_eval:
                msg += f" | val {val_loss:.4f}"
            if do_log:
                msg += f" | grad {grad_norm:.2f}"
            msg += f" | {step_ms:.0f} ms/step | {tokens_per_step / step_ms * 1000:,.0f} tok/s"
            print(msg, flush=True)
            if not math.isfinite(train_loss):
                print("loss가 inf 또는 nan입니다. 학습률을 낮춰 다시 시작하세요.")
                break

        # 6) 체크포인트 (5.2)
        if args.ckpt_interval and (step + 1) % args.ckpt_interval == 0:
            save_checkpoint(model, opt, step + 1, run_dir / f"ckpt_{step + 1}.pt")
            save_checkpoint(model, opt, step + 1, run_dir / "ckpt_latest.pt")

    save_checkpoint(model, opt, step + 1, run_dir / "ckpt_final.pt")
    log_f.close()
    total_min = (time.perf_counter() - t_start) / 60
    peak = torch.cuda.max_memory_allocated() / 2**30 if args.device.startswith("cuda") else 0
    summary = {"steps": step + 1, "minutes": round(total_min, 2), "peak_mem_GiB": round(peak, 2),
               "params": n_params, "precision": "bf16" if use_bf16 else "fp32"}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()