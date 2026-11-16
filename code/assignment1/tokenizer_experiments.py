"""2.7 Problem (tokenizer_experiments): TinyStories 쪽 (a) (c) (d).

assignment1-basics 루트에 두고 실행한다.

  (a) 압축률:   uv run python tokenizer_experiments.py ratio
  (c) 처리량:   uv run python tokenizer_experiments.py throughput
  (d) 인코딩:   uv run python tokenizer_experiments.py encode --input data/TinyStoriesV2-GPT4-valid.txt --out data/ts_valid.npy
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer

SPECIAL = "<|endoftext|>"
VOCAB = "out/ts_bpe/vocab.pkl"
MERGES = "out/ts_bpe/merges.pkl"
VALID = "data/TinyStoriesV2-GPT4-valid.txt"


def load_tokenizer():
    return Tokenizer.from_files(VOCAB, MERGES, [SPECIAL])


def ratio(args):
    """(a) 문서 10개를 뽑아 압축률(바이트/토큰) 계산"""
    text = Path(args.input).read_text(encoding="utf-8")
    docs = [d for d in text.split(SPECIAL) if d.strip()]
    random.seed(0)  # 다시 돌려도 같은 문서가 뽑히게
    sample = random.sample(docs, 10)

    tok = load_tokenizer()
    n_bytes = sum(len(d.encode("utf-8")) for d in sample)
    n_tokens = sum(len(tok.encode(d)) for d in sample)
    print(f"문서 {len(sample)}개, {n_bytes} 바이트, {n_tokens} 토큰")
    print(f"압축률: {n_bytes / n_tokens:.2f} 바이트/토큰")


def throughput(args):
    """(c) 처리량(바이트/초)과 Pile(825GB) 인코딩 예상 시간"""
    text = Path(args.input).read_text(encoding="utf-8")
    n_bytes = len(text.encode("utf-8"))

    tok = load_tokenizer()  # 캐시가 빈 상태에서 시작
    t0 = time.perf_counter()
    ids = tok.encode(text)
    dt = time.perf_counter() - t0

    bps = n_bytes / dt
    pile_hours = 825e9 / bps / 3600
    print(f"{n_bytes / 1e6:.1f} MB를 {dt:.1f}초에 인코딩 ({len(ids)} 토큰)")
    print(f"처리량: {bps / 1e6:.2f} MB/s")
    print(f"Pile 825GB 예상: {pile_hours:.0f}시간 (약 {pile_hours / 24:.1f}일)")


def encode(args):
    """(d) 파일 전체를 토큰 ID로 바꿔 uint16 배열로 저장"""
    tok = load_tokenizer()
    assert max(tok.vocab) < 2**16, "어휘가 65536개를 넘으면 uint16에 못 담음"

    n_bytes = Path(args.input).stat().st_size
    t0 = time.perf_counter()
    with open(args.input, encoding="utf-8") as f:
        # encode_iterable이 한 줄씩 ID를 내놓으면 fromiter가 바로 배열에 담는다
        ids = np.fromiter(tok.encode_iterable(f), dtype=np.uint16)
    dt = time.perf_counter() - t0

    np.save(args.out, ids)
    summary = {
        "input": args.input,
        "output": args.out,
        "bytes": n_bytes,
        "tokens": int(ids.size),
        "bytes_per_token": round(n_bytes / ids.size, 3),
        "elapsed_min": round(dt / 60, 2),
        "throughput_MBps": round(n_bytes / dt / 1e6, 3),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["ratio", "throughput", "encode"])
    p.add_argument("--input", default=VALID)
    p.add_argument("--out")
    args = p.parse_args()
    {"ratio": ratio, "throughput": throughput, "encode": encode}[args.task](args)