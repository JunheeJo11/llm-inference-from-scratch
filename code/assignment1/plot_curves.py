"""11/16 학습 3종의 손실 곡선

위치: assignment1-basics/scratch/plot_curves.py
실행: uv run --with matplotlib python scratch/plot_curves.py
결과: runs/curves.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로만 그림 (WSL에서 창을 띄우지 않음)
import matplotlib.pyplot as plt

RUNS = ["ts_L2", "ts_base", "ts_L8"]
TARGET = 1.45  # 핸드아웃 7.2.1 목표 검증 loss

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for run in RUNS:
    rows = list(csv.DictReader(open(Path("runs") / run / "log.csv")))
    steps = [int(r["step"]) for r in rows]
    train = [float(r["train_loss"]) for r in rows]
    # val_loss는 평가한 스텝에만 값이 있고 나머지 줄은 비어 있음
    vs = [(int(r["step"]), float(r["val_loss"])) for r in rows if r["val_loss"]]
    line, = axes[0].plot(steps, train, lw=1, alpha=0.8, label=f"{run} train")
    axes[0].plot([s for s, _ in vs], [v for _, v in vs], "o-", ms=3, color=line.get_color(), label=f"{run} val")
    axes[1].plot([s for s, _ in vs], [v for _, v in vs], "o-", ms=3, label=f"{run} val (final {vs[-1][1]:.3f})")

for ax in axes:
    ax.axhline(TARGET, ls="--", c="gray", lw=1)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_title("train and val loss")
axes[1].set_title("val loss, zoom (dashed = handout target 1.45)")
axes[1].set_ylim(1.2, 2.2)
fig.tight_layout()
fig.savefig("runs/curves.png", dpi=120)
print("저장: runs/curves.png")