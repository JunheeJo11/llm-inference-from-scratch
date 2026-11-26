"""11/26 배치 곡선과 정적 배치 대 continuous batching 측정

위치: assignment1-basics/scratch/bench_batch.py
실행 (과제 폴더 맨 위에서): uv run python scratch/bench_batch.py configs/bench_batch.toml
결과: runs/batch/<name>_<시각>.json 과 .png

실험 A, 배치 크기 곡선 (모든 슬롯이 꽉 찬 상태, 길이 같음)
  배치 B마다
  - decode 한 스텝 시간 (= 배치 B에서의 TPOT)과 처리량 B / 스텝 시간
  - 배치 prefill 시간 (= 배치 B로 함께 들어온 요청들의 TTFT), 프롬프트 길이별
  - 메모리: 캐시 크기, decode 실행 중 최대 추가 메모리, 긴 prefill 중 최대 추가 메모리
실험 B, 길이가 제각각인 요청 N개를 슬롯 수별로
  - 정적 배치 / continuous (먼저 온 순서) / continuous (긴 요청 먼저)
  - 처리량, TTFT 평균과 p50, p99, 요청이 끝날 때까지 걸린 시간 평균, 토큰을 만든 칸 비율
"""
import json
import random
import statistics
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

import torch

import cs336_basics.batch_infer as bi
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

EOT = "<|endoftext|>"
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
STORY = ("Once upon a time, there was a little girl named Lily. She loved to play in the park with her dog, Max. "
         "One day, Lily saw a big red ball under a tree. She ran to the ball and kicked it high into the sky. "
         "Max barked and jumped. The ball fell into the pond, and Lily was sad. Then a kind old man came by "
         "with a long stick. He helped Lily get the ball back. Lily said thank you and gave him a big hug. ")


def load_model(run_dir: Path, device: str, dtype):
    cfg = json.loads((run_dir / "config.json").read_text())
    model = TransformerLM(
        cfg["vocab_size"], cfg["context_length"], cfg["d_model"], cfg["num_layers"],
        cfg["num_heads"], cfg["d_ff"], cfg["rope_theta"], device=device,
    )
    ckpt = torch.load(run_dir / "ckpt_final.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(dtype)
    model.eval()
    return model


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def pct(xs, p):
    """p번째 백분위수 (p50 = 중앙값). 값이 하나면 그 값"""
    if len(xs) < 2:
        return xs[0]
    return statistics.quantiles(xs, n=100)[p - 1]


def ms_stats(xs):
    return statistics.mean(xs) * 1e3, (statistics.stdev(xs) * 1e3 if len(xs) > 1 else 0.0)


# ------------------------------------------------------------------ 실험 A
@torch.no_grad()
def curve(model, story, c):
    rows = []
    mib = 2 ** 20
    print(f"\n[A] 배치 크기 곡선: decode {c['decode_steps']}스텝 (프롬프트 {c['prompt_len']}), "
          f"워밍업 {c['warmup']}, 반복 {c['repeats']}")
    head = (f"{'B':>4} | {'스텝 ms':>13} | {'tok/s':>7} | "
            + " | ".join(f"{'TTFT P' + str(P) + ' ms':>15}" for P in c["prefill_lens"])
            + f" | {'캐시 MiB':>8} {'decode 최대':>10} {'prefill 최대':>11}")
    print(head)
    print("-" * len(head))

    for B in c["batch_sizes"]:
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()               # 모델 가중치 등 이미 잡힌 메모리
        cache = bi.SlotKVCache(model, B)
        slots = list(range(B))
        active = [True] * B

        # decode 한 스텝 시간
        torch.cuda.reset_peak_memory_stats()
        prompt = story[:c["prompt_len"]]
        step_times = []
        for rep in range(c["warmup"] + c["repeats"]):
            cache.reset()
            tokens = torch.argmax(bi.prefill(model, cache, slots, [prompt] * B), dim=-1)
            tokens.tolist()
            sync()
            t0 = time.perf_counter()
            for _ in range(c["decode_steps"]):
                tokens = torch.argmax(bi.decode_step(model, cache, tokens, active), dim=-1)
                tokens.tolist()                            # 실제 스케줄러처럼 스텝마다 토큰을 CPU로
            sync()
            if rep >= c["warmup"]:
                step_times.append((time.perf_counter() - t0) / c["decode_steps"])
        decode_peak = torch.cuda.max_memory_allocated() - base

        # 배치 prefill 시간 (프롬프트 길이별)
        ttft = {}
        prefill_peak = 0
        for P in c["prefill_lens"]:
            prompt = story[:P]
            torch.cuda.reset_peak_memory_stats()
            ts = []
            for rep in range(c["warmup"] + c["repeats"]):
                cache.reset()
                sync()
                t0 = time.perf_counter()
                torch.argmax(bi.prefill(model, cache, slots, [prompt] * B), dim=-1).tolist()
                if rep >= c["warmup"]:
                    ts.append(time.perf_counter() - t0)
            prefill_peak = max(prefill_peak, torch.cuda.max_memory_allocated() - base)
            ttft[P] = ts

        m, s = ms_stats(step_times)
        row = dict(B=B, step_ms=m, step_ms_std=s, tok_s=B / statistics.mean(step_times),
                   ttft_ms={P: ms_stats(v)[0] for P, v in ttft.items()},
                   ttft_ms_std={P: ms_stats(v)[1] for P, v in ttft.items()},
                   prefill_tok_s={P: B * P / statistics.mean(v) for P, v in ttft.items()},
                   cache_mib=cache.nbytes() / mib, decode_peak_mib=decode_peak / mib,
                   prefill_peak_mib=prefill_peak / mib)
        rows.append(row)
        print(f"{B:>4} | {m:6.2f} ± {s:4.2f} | {row['tok_s']:7.0f} | "
              + " | ".join(f"{row['ttft_ms'][P]:7.2f} ± {row['ttft_ms_std'][P]:5.2f}" for P in c["prefill_lens"])
              + f" | {row['cache_mib']:8.0f} {row['decode_peak_mib']:10.0f} {row['prefill_peak_mib']:11.0f}")
        del cache
    return rows


# ------------------------------------------------------------------ 실험 B
def make_workload(story, w):
    rng = random.Random(w["seed"])
    prompts, max_new = [], []
    for _ in range(w["num_requests"]):
        P = rng.randint(w["prompt_min"], w["prompt_max"])
        n = rng.randint(w["gen_min"], w["gen_max"])
        off = rng.randrange(0, len(story) - P)
        prompts.append(story[off:off + P])
        max_new.append(n)
    return prompts, max_new


def summarize(stats, max_new):
    tf, td = stats["t_first"], stats["t_done"]
    return dict(total_s=stats["total_s"], tok_s=sum(max_new) / stats["total_s"],
                ttft_mean_ms=statistics.mean(tf) * 1e3, ttft_p50_ms=pct(tf, 50) * 1e3,
                ttft_p99_ms=pct(tf, 99) * 1e3, done_mean_ms=statistics.mean(td) * 1e3,
                useful_pct=100 * stats["useful_row_steps"] / stats["row_steps"],
                decode_steps=stats["decode_steps"])


def workload_bench(model, story, w):
    prompts, max_new = make_workload(story, w)
    lpt = sorted(range(len(prompts)), key=lambda i: -max_new[i])   # 긴 요청 먼저
    lpt_prompts = [prompts[i] for i in lpt]
    lpt_new = [max_new[i] for i in lpt]
    Ps = [len(p) for p in prompts]
    print(f"\n[B] 요청 {len(prompts)}개, P {min(Ps)}~{max(Ps)} (평균 {statistics.mean(Ps):.0f}), "
          f"n {min(max_new)}~{max(max_new)} (평균 {statistics.mean(max_new):.0f}), 생성 토큰 합 {sum(max_new)}, "
          f"반복 {w['repeats']}")
    head = (f"{'슬롯':>4} {'방식':>14} | {'tok/s':>12} | {'TTFT 평균':>9} {'p50':>7} {'p99':>7} ms | "
            f"{'완료 평균 ms':>11} | {'스텝':>5} {'토큰 칸':>6}")
    print(head)
    print("-" * len(head))

    methods = {
        "static": lambda S: bi.run_static(model, prompts, max_new, S),
        "continuous": lambda S: bi.run_continuous(model, prompts, max_new, S),
        "cont_longest": lambda S: bi.run_continuous(model, lpt_prompts, lpt_new, S),
    }
    rows = []
    for S in w["slot_counts"]:
        bi.run_continuous(model, prompts[:2 * S], max_new[:2 * S], S)   # 워밍업 (결과 버림)
        for name, fn in methods.items():
            runs = []
            for _ in range(w["repeats"]):
                outs, st = fn(S)
                runs.append(summarize(st, lpt_new if name == "cont_longest" else max_new))
            tok = [r["tok_s"] for r in runs]
            avg = {k: statistics.mean(r[k] for r in runs) for k in runs[0]}
            row = dict(slots=S, method=name, **avg, tok_s_std=statistics.stdev(tok) if len(tok) > 1 else 0.0)
            rows.append(row)
            print(f"{S:>4} {name:>14} | {row['tok_s']:5.0f} ± {row['tok_s_std']:4.0f} | "
                  f"{row['ttft_mean_ms']:9.0f} {row['ttft_p50_ms']:7.0f} {row['ttft_p99_ms']:7.0f}    | "
                  f"{row['done_mean_ms']:11.0f} | {row['decode_steps']:5.0f} {row['useful_pct']:5.0f}%")
    return rows


# ------------------------------------------------------------------ 그림
def plot(curve_rows, wl_rows, conf, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = ["#257de0", "#f47230", "#00b894", "#f2a600"]       # 고정 순서 색 (파랑, 주황, 청록, 노랑)
    ink, grid = "#3a3a3a", "#e3e3e3"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": grid, "axes.labelcolor": ink,
                         "xtick.color": ink, "ytick.color": ink, "axes.spines.top": False,
                         "axes.spines.right": False})
    for fam in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic"]:
        if any(fam == f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = fam
            break

    Bs = [r["B"] for r in curve_rows]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    line = dict(linewidth=2, marker="o", markersize=5)

    a = ax[0, 0]
    a.plot(Bs, [r["tok_s"] for r in curve_rows], color=C[0], **line)
    a.set(title="decode 처리량", xlabel="배치 크기 B", ylabel="tok/s", xscale="log", yscale="log")

    a = ax[0, 1]
    a.plot(Bs, [r["step_ms"] for r in curve_rows], color=C[0], label="decode 한 스텝", **line)
    for i, P in enumerate(conf["curve"]["prefill_lens"]):
        a.plot(Bs, [r["ttft_ms"][P] for r in curve_rows], color=C[i + 1], label=f"배치 prefill P={P}", **line)
    a.set(title="스텝 시간과 배치 prefill 시간", xlabel="배치 크기 B", ylabel="ms", xscale="log", yscale="log")
    a.legend(frameon=False)

    a = ax[1, 0]
    a.plot(Bs, [r["cache_mib"] for r in curve_rows], color=C[0], label="캐시 (미리 잡은 크기)", **line)
    a.plot(Bs, [r["decode_peak_mib"] for r in curve_rows], color=C[1], label="decode 중 최대 추가", **line)
    a.plot(Bs, [r["prefill_peak_mib"] for r in curve_rows], color=C[2], label="prefill 중 최대 추가", **line)
    a.set(title="메모리 (가중치 제외)", xlabel="배치 크기 B", ylabel="MiB", xscale="log", yscale="log")
    a.legend(frameon=False)

    a = ax[1, 1]
    slots = conf["workload"]["slot_counts"]
    names = [("static", "정적"), ("continuous", "continuous"), ("cont_longest", "continuous, 긴 요청 먼저")]
    wdt = 0.26
    for i, (key, label) in enumerate(names):
        ys = [next(r["tok_s"] for r in wl_rows if r["slots"] == S and r["method"] == key) for S in slots]
        xs = [j + (i - 1) * wdt for j in range(len(slots))]
        a.bar(xs, ys, width=wdt - 0.02, color=C[i], label=label)
    a.set_xticks(range(len(slots)), [f"슬롯 {S}" for S in slots])
    a.set(title="길이가 제각각인 요청: 처리량", ylabel="tok/s")
    a.legend(frameon=False)

    from matplotlib import ticker
    for a in [ax[0, 0], ax[0, 1], ax[1, 0]]:              # 로그 축 눈금을 1, 2, 4 ... 그대로 숫자로
        a.set_xticks(Bs, [str(b) for b in Bs])
        a.xaxis.set_minor_formatter(ticker.NullFormatter())
        a.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        a.yaxis.set_minor_formatter(ticker.NullFormatter())
    for a in ax.flat:
        a.grid(True, color=grid, linewidth=0.8)
        a.set_axisbelow(True)
    fig.suptitle(f"{conf['run']}, {conf['dtype']}, greedy", color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    if len(sys.argv) != 2:
        print("사용법: uv run python scratch/bench_batch.py configs/<설정>.toml")
        sys.exit(1)
    cfg_path = Path(sys.argv[1])
    with open(cfg_path, "rb") as f:
        conf = tomllib.load(f)

    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(Path("runs") / conf["run"], conf.get("device", "cuda"), DTYPES[conf["dtype"]])
    tok = Tokenizer.from_files("out/ts_bpe/vocab.pkl", "out/ts_bpe/merges.pkl", special_tokens=[EOT])
    story = tok.encode(STORY * 8)
    env = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
               tf32=torch.backends.cuda.matmul.allow_tf32, time=datetime.now().isoformat(timespec="seconds"))
    print(f"설정 {cfg_path} ({conf['name']}): {conf['run']}, {conf['dtype']}, greedy")
    print(f"환경: {env['gpu']}, torch {env['torch']}, TF32 {env['tf32']}")

    curve_rows = curve(model, story, conf["curve"])
    wl_rows = workload_bench(model, story, conf["workload"])
    print(f"\nRoPE 방식: {bi.ROPE_MODE}")

    out_dir = Path(conf["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{conf['name']}_{datetime.now():%Y%m%d_%H%M%S}"
    stem.with_suffix(".json").write_text(json.dumps(
        dict(config=conf, config_file=str(cfg_path), env=env, rope_mode=bi.ROPE_MODE,
             curve=curve_rows, workload=wl_rows), ensure_ascii=False, indent=2))
    plot(curve_rows, wl_rows, conf, stem.with_suffix(".png"))
    print(f"저장: {stem}.json, {stem}.png")


if __name__ == "__main__":
    main()
