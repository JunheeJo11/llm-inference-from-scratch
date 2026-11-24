"""11/24 TTFT와 TPOT 측정. 측정 조건은 설정 파일(TOML)로 뺀다

위치: assignment1-basics/scratch/measure_latency.py
설정: assignment1-basics/configs/*.toml
실행 (과제 폴더 맨 위에서):
    uv run python scratch/measure_latency.py configs/latency_base.toml

용어
  TTFT (time to first token): 생성 시작부터 첫 토큰이 손에 들어올 때까지. prefill 한 번 + 첫 토큰 뽑기
  TPOT (time per output token): 첫 토큰 이후 토큰 하나당 평균 시간 = (마지막 토큰 시각 - 첫 토큰 시각) / (n - 1)
  ITL  (inter-token latency): 이웃한 두 토큰 사이 간격 하나하나. TPOT는 이것들의 평균

결과: 화면 표 + <out_dir>/<name>_<날짜시각>.json (설정, 환경, 조건별 요약, 실행마다 원래 숫자)
"""
import json
import statistics
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

import torch

from cs336_basics.kv_cache import KVcache, forward_with_cache
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

EOT = "<|endoftext|>"
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
# 프롬프트 길이를 정확히 맞추려고 이 글을 반복해서 붙인 뒤 앞에서 L토큰만 자른다
FILLER = ("Once upon a time, there was a little girl named Lily. She loved to play outside "
          "in the park with her friends. One day, she saw a big red ball under a tree. ")


def load_model(run_dir: Path, device: str, dtype: torch.dtype):
    cfg = json.loads((run_dir / "config.json").read_text())
    model = TransformerLM(
        cfg["vocab_size"], cfg["context_length"], cfg["d_model"], cfg["num_layers"],
        cfg["num_heads"], cfg["d_ff"], cfg["rope_theta"], device=device,
    )
    ckpt = torch.load(run_dir / "ckpt_final.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(dtype)
    model.eval()
    return model, cfg


def make_prompt(tok: Tokenizer, length: int) -> list[int]:
    """정확히 length토큰짜리 프롬프트"""
    ids: list[int] = []
    k = 1
    while len(ids) < length:
        ids = tok.encode(FILLER * k)
        k *= 2
    return ids[:length]


def sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def timed_generate(model, prompt_ids: list[int], n: int, use_cache: bool, device: str):
    """greedy로 n토큰을 만들며 토큰마다 '손에 들어온 시각'(시작 기준 초)을 기록한다

    int(next_id)가 GPU 값을 CPU로 가져오면서 GPU가 끝날 때까지 기다리므로,
    그 직후에 찍은 시각은 그 토큰이 실제로 나온 시각이다
    """
    # 캐시는 시계 밖에서 미리 만든다 (vLLM도 서버 시작 때 캐시 메모리를 미리 잡아 둔다)
    cache = KVcache.for_model(model) if use_cache else None
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, P)
    out: list[int] = []
    times: list[float] = []

    sync(device)
    t0 = time.perf_counter()
    # prefill: 프롬프트 전체를 한 번에
    if use_cache:
        logits = forward_with_cache(model, ids, cache, last_only=True)  # (1, 1, V)
    else:
        logits = model(ids)[:, -1:]
    for step in range(n):
        next_id = torch.argmax(logits[0, -1])
        out.append(int(next_id))                 # 여기서 GPU 기다림 (동기화)
        times.append(time.perf_counter() - t0)   # 이 토큰이 나온 시각
        if step == n - 1:
            break
        if use_cache:
            logits = forward_with_cache(model, next_id.view(1, 1), cache)
        else:
            ids = torch.cat([ids, next_id.view(1, 1)], dim=1)  # 캐시 없음: 문장 전체를 다시 넣음
            logits = model(ids)[:, -1:]
    return out, times


def percentile(xs: list[float], p: int) -> float:
    """p번째 백분위수. statistics.quantiles는 1~99번째 경계 99개를 돌려준다"""
    return statistics.quantiles(xs, n=100)[p - 1]


def main():
    if len(sys.argv) != 2:
        print("사용법: uv run python scratch/measure_latency.py configs/<설정>.toml")
        sys.exit(1)
    cfg_path = Path(sys.argv[1])
    with open(cfg_path, "rb") as f:  # tomllib은 바이트로 열어야 함
        conf = tomllib.load(f)

    device = conf.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model, mcfg = load_model(Path("runs") / conf["run"], device, DTYPES[conf["dtype"]])
    tok = Tokenizer.from_files("out/ts_bpe/vocab.pkl", "out/ts_bpe/merges.pkl", special_tokens=[EOT])
    ctx = mcfg["context_length"]
    gpu = torch.cuda.get_device_name() if device.startswith("cuda") else "cpu"
    env = dict(gpu=gpu, torch=torch.__version__, tf32=torch.backends.cuda.matmul.allow_tf32,
               time=datetime.now().isoformat(timespec="seconds"))
    print(f"설정 {cfg_path} ({conf['name']}): {conf['run']}, {conf['dtype']}, 배치 1, greedy, "
          f"워밍업 {conf['warmup']}, 반복 {conf['repeats']}")
    print(f"환경: {gpu}, torch {env['torch']}, TF32 {env['tf32']}\n")

    head = (f"{'방식':>7} {'P':>4} {'n':>4} | {'TTFT ms':>14} | {'TPOT ms':>14} | "
            f"{'ITL p50':>7} {'p90':>6} {'p99':>6} | {'tok/s':>6}")
    print(head)
    print("-" * len(head))

    rows = []
    for mode in conf["modes"]:
        use_cache = mode == "cache"
        for P in conf["prompt_lens"]:
            for n in conf["gen_lens"]:
                if P + n > ctx:
                    print(f"{mode:>7} {P:>4} {n:>4} | 건너뜀: P + n = {P + n} > context {ctx}")
                    continue
                prompt = make_prompt(tok, P)
                for _ in range(conf["warmup"]):
                    timed_generate(model, prompt, n, use_cache, device)
                ttfts, tpots, itls, e2es = [], [], [], []
                for _ in range(conf["repeats"]):
                    _, times = timed_generate(model, prompt, n, use_cache, device)
                    ttfts.append(times[0])
                    tpots.append((times[-1] - times[0]) / (n - 1))
                    itls += [b - a for a, b in zip(times, times[1:])]  # 이웃 토큰 간격들
                    e2es.append(times[-1])
                ms = 1e3
                row = dict(
                    mode=mode, prompt_len=P, gen_len=n,
                    ttft_ms=statistics.mean(ttfts) * ms, ttft_sd=statistics.stdev(ttfts) * ms,
                    tpot_ms=statistics.mean(tpots) * ms, tpot_sd=statistics.stdev(tpots) * ms,
                    itl_p50=percentile(itls, 50) * ms, itl_p90=percentile(itls, 90) * ms,
                    itl_p99=percentile(itls, 99) * ms,
                    tok_per_s=n / statistics.mean(e2es),
                    raw=dict(ttft_ms=[t * ms for t in ttfts], tpot_ms=[t * ms for t in tpots]),
                )
                rows.append(row)
                print(f"{mode:>7} {P:>4} {n:>4} | {row['ttft_ms']:>7.2f} ±{row['ttft_sd']:>5.2f} | "
                      f"{row['tpot_ms']:>7.2f} ±{row['tpot_sd']:>5.2f} | "
                      f"{row['itl_p50']:>7.2f} {row['itl_p90']:>6.2f} {row['itl_p99']:>6.2f} | "
                      f"{row['tok_per_s']:>6.0f}")

    out_dir = Path(conf.get("out_dir", "runs/latency"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{conf['name']}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps(dict(config=conf, config_file=str(cfg_path), env=env, rows=rows),
                                   ensure_ascii=False, indent=2))
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
