"""6장과 7.2 Problem (generate): 학습한 모델로 글 생성

위치: assignment1-basics/generate.py (train.py 옆)

예시
  uv run python generate.py --run ts_base
  uv run python generate.py --run ts_L8 --temperature 0.7 --top_p 0.9

결과: 화면 출력 + samples/<run>.md 끝에 이어 쓰기
"""
import argparse
import json
import time
from pathlib import Path

import torch

from cs336_basics.decoding import generate
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

EOT = "<|endoftext|>"


def load_model(run_dir: Path, device: str):
    """학습 때 저장한 config.json으로 같은 모양의 모델을 만들고 ckpt_final.pt의 가중치를 넣는다"""
    cfg = json.loads((run_dir / "config.json").read_text())
    model = TransformerLM(
        cfg["vocab_size"], cfg["context_length"], cfg["d_model"], cfg["num_layers"],
        cfg["num_heads"], cfg["d_ff"], cfg["rope_theta"], device=device,
    )
    ckpt = torch.load(run_dir / "ckpt_final.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])  # 옵티마이저 상태는 생성에 필요 없어서 안 불러옴
    model.eval()
    return model, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="ts_base")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--n", type=int, default=3)  # 같은 설정으로 몇 개 뽑을지
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    run_dir = Path("runs") / args.run
    model, cfg = load_model(run_dir, args.device)
    tok = Tokenizer.from_files("out/ts_bpe/vocab.pkl", "out/ts_bpe/merges.pkl", special_tokens=[EOT])
    eos_id = tok.encode(EOT)[0]
    prompt_ids = tok.encode(args.prompt)
    cuda = args.device.startswith("cuda")

    lines = [f"## {args.run} | T={args.temperature}, top_p={args.top_p}, seed={args.seed}, "
             f"max {args.max_new_tokens}, 프롬프트 \"{args.prompt}\"", ""]
    for i in range(args.n):
        if cuda:
            torch.cuda.synchronize()  # 앞서 걸린 GPU 작업이 끝난 뒤 시계를 잼
        t0 = time.perf_counter()
        new_ids = generate(model, prompt_ids, args.max_new_tokens, cfg["context_length"],
                           args.temperature, args.top_p, eos_id)
        if cuda:
            torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        stop = "eos" if len(new_ids) < args.max_new_tokens else "max"
        info = f"{len(new_ids)}토큰, {sec:.2f}초, {len(new_ids) / max(sec, 1e-9):.0f} tok/s, 종료 {stop}"
        text = args.prompt + tok.decode(new_ids)
        print(f"--- [{i + 1}] {info}\n{text}\n")
        lines += [f"**[{i + 1}]** ({info})", "", text, ""]

    out_dir = Path("samples")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"{args.run}.md", "a") as f:  # "a": 실행할 때마다 끝에 이어 씀
        f.write("\n".join(lines) + "\n")
    print(f"저장: {out_dir / (args.run + '.md')}")


if __name__ == "__main__":
    main()