"""11/27 같은 조건 비교 2단계: 내 모델(ts_base)을 vLLM에 올려 같은 요청으로 잰다

위치: ~/llm-study/vllm-lab/compare_same_cond.py   (vllm-lab venv에서 실행)

모드 두 개
  python compare_same_cond.py --check-hf
      vLLM을 띄우지 않고 transformers 로만 변환이 맞는지 본다 (fp32, 몇 초).
      내 정답 토큰을 그대로 넣었을 때(teacher forcing) 각 자리의 1등 토큰이 정답 다음 토큰과 같은지.
  python compare_same_cond.py --slots 16 [--eager] [--grid] [--dtype bfloat16]
      vLLM 엔진을 한 번 띄워서
        [A] 정답 비교: 앞 32개 요청을 greedy 로 생성해 내 정답과 몇 개가 같은지
        [B] (--grid) 배치 1 길이 격자: P x n 별 TTFT 와 TPOT (11/26 latency_grid 와 같은 격자)
        [C] 처리량: 요청 256개를 한 번에 넣고 tok/s (max_num_seqs = 슬롯 수)
      결과는 compare/vllm_<설정>.json

용어
  max_num_seqs  : 한 스텝에 같이 돌리는 요청 수의 상한. 내 코드의 "슬롯 수"와 같은 역할
  --eager       : CUDA graph 를 끄고(enforce_eager) 커널을 매번 하나씩 보낸다. 내 구현과 같은 방식
  --no-prefix-cache : prefix caching 을 끈다. 켜 두면 같은 프롬프트를 반복할 때 prefill 을 건너뛰어서
                  내 구현(캐시 재사용 없음)과 조건이 달라진다. 11/27 수정: 같은 조건 비교에는 이걸 켤 것
  skip_tokenizer_init : 토크나이저 없이 토큰 번호만 주고받기. 내 BPE 는 HF 형식이 아니라서
"""
import argparse
import json
import statistics
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ensure_safetensors(model_dir: Path):
    """export 쪽에 safetensors 가 없어서 .bin 으로 저장됐으면 여기서 바꿔 둔다"""
    st, binf = model_dir / "model.safetensors", model_dir / "pytorch_model.bin"
    if st.exists() or not binf.exists():
        return
    import torch
    from safetensors.torch import save_file
    sd = torch.load(binf, map_location="cpu", weights_only=True)
    save_file({k: v.contiguous() for k, v in sd.items()}, str(st))
    binf.rename(binf.with_suffix(".bin.bak"))       # 둘 다 있으면 헷갈리니 이름만 바꿔 둠
    print(f"safetensors 로 바꿈: {st}")


def check_hf(model_dir: Path, data):
    """vLLM 없이 transformers 로 변환 확인 (fp32, teacher forcing)"""
    import torch
    from transformers import AutoModelForCausalLM
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).to(dev).eval()
    hit = tot = 0
    first_bad = []
    for i, r in enumerate(x for x in data["requests"] if "ref" in x):
        seq = r["prompt"] + r["ref"]
        with torch.no_grad():
            logits = m(torch.tensor([seq], device=dev)).logits[0]
        P = len(r["prompt"])
        pred = logits[P - 1:P - 1 + len(r["ref"])].argmax(-1).tolist()   # 각 자리에서 다음 토큰 예측
        same = [a == b for a, b in zip(pred, r["ref"])]
        hit += sum(same)
        tot += len(same)
        if not all(same):
            first_bad.append((i, same.index(False)))
    print(f"[HF 확인] teacher forcing 1등 토큰 일치 {hit}/{tot} ({100 * hit / tot:.2f}%)")
    if first_bad:
        print(f"  안 맞는 요청 (번호, 처음 갈라진 자리): {first_bad[:10]}")
        print("  거의 다 틀리면 RoPE 순서 바꾸기 문제일 가능성이 큼 -> 결과를 붙여 주세요")
    else:
        print("  전부 일치: 변환이 맞아요. vLLM 측정으로 넘어가면 돼요")


def gen(llm, SP, prompts, ns):
    params = [SP(temperature=0.0, max_tokens=n, ignore_eos=True, detokenize=False) for n in ns]
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
    return [list(o.outputs[0].token_ids) for o in outs]


def timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ts_base")
    ap.add_argument("--check-hf", action="store_true")
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--no-prefix-cache", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--grid-repeats", type=int, default=5)
    args = ap.parse_args()

    model_dir = HERE / "models" / f"{args.run}_hf"
    data = json.loads((HERE / "compare" / f"requests_{args.run}.json").read_text())
    reqs = data["requests"]
    ensure_safetensors(model_dir)

    if args.check_hf:
        check_hf(model_dir, data)
        return

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    tag = (f"{args.dtype}_s{args.slots}{'_eager' if args.eager else ''}{'_grid' if args.grid else ''}"
           f"{'_nopc' if args.no_prefix_cache else ''}")
    print(f"vllm {vllm.__version__}, torch {torch.__version__}, 설정 {tag}")
    t0 = time.perf_counter()
    llm = LLM(
        model=str(model_dir),
        skip_tokenizer_init=True,
        dtype=args.dtype,
        max_model_len=data["context_length"],   # 256
        max_num_seqs=args.slots,
        enforce_eager=args.eager,
        gpu_memory_utilization=0.5,
        seed=0,
        enable_prefix_caching=not args.no_prefix_cache,
    )
    start_s = time.perf_counter() - t0
    print(f"엔진 시작 {start_s:.1f} s")
    res = {"vllm": vllm.__version__, "torch": torch.__version__, "dtype": args.dtype,
           "slots": args.slots, "eager": args.eager, "prefix_caching": not args.no_prefix_cache,
           "start_s": start_s}

    # [A] 정답 비교
    refd = [r for r in reqs if "ref" in r]
    outs = gen(llm, SamplingParams, [r["prompt"] for r in refd], [r["n"] for r in refd])
    same, splits = 0, []
    for i, (o, r) in enumerate(zip(outs, refd)):
        if o == r["ref"]:
            same += 1
        else:
            splits.append((i, next(j for j, (a, b) in enumerate(zip(o, r["ref"])) if a != b)))
    print(f"[A] 내 정답(fp32)과 완전 일치 {same}/{len(refd)}  갈라진 곳 (요청, 자리): {splits[:10]}")
    res["match"] = {"same": same, "total": len(refd), "splits": splits}

    # [B] 배치 1 길이 격자: TTFT = max_tokens 1 의 시간, TPOT = (n개 시간 - 1개 시간) / (n - 1)
    if args.grid:
        pool = [t for r in reqs for t in r["prompt"]]
        grid = []
        print(f"[B] 배치 1 격자 (워밍업 1, 반복 {args.grid_repeats}, 중앙값)")
        print(f"{'P':>4} {'n':>4} | {'TTFT ms':>8} | {'TPOT ms':>8}")
        for P in [4, 32, 64, 128, 192, 240]:
            for n in [16, 64, 128]:
                if P + n > data["context_length"]:
                    continue
                p = [pool[:P]]
                one = lambda: gen(llm, SamplingParams, p, [1])
                full = lambda: gen(llm, SamplingParams, p, [n])
                one(), full()                                   # 워밍업
                t1s, tps = [], []
                for _ in range(args.grid_repeats):
                    t1, tn = timed(one), timed(full)
                    t1s.append(t1)
                    tps.append((tn - t1) / (n - 1))
                row = {"P": P, "n": n, "ttft_ms": statistics.median(t1s) * 1e3,
                       "tpot_ms": statistics.median(tps) * 1e3}
                grid.append(row)
                print(f"{P:>4} {n:>4} | {row['ttft_ms']:>8.2f} | {row['tpot_ms']:>8.3f}")
        res["grid"] = grid
        print(f"  중앙값: TTFT {statistics.median(g['ttft_ms'] for g in grid):.2f} ms, "
              f"TPOT {statistics.median(g['tpot_ms'] for g in grid):.3f} ms")

    # [C] 처리량: 256개를 한 번에 (모두 시각 0 도착, 11/26 실험 B와 같은 가정)
    prompts, ns = [r["prompt"] for r in reqs], [r["n"] for r in reqs]
    total = sum(ns)
    outs = gen(llm, SamplingParams, prompts, ns)                    # 워밍업
    assert sum(len(o) for o in outs) == total, "생성 토큰 수가 요청과 다름"
    ts = [timed(lambda: gen(llm, SamplingParams, prompts, ns)) for _ in range(args.repeats)]
    tps = [total / t for t in ts]
    res["throughput"] = {"sec": ts, "tok_s_mean": statistics.mean(tps),
                         "tok_s_std": statistics.stdev(tps) if len(tps) > 1 else 0.0, "tokens": total}
    print(f"[C] 처리량 슬롯 {args.slots}: {res['throughput']['tok_s_mean']:,.0f} ± "
          f"{res['throughput']['tok_s_std']:,.0f} tok/s (256개, {total:,}토큰, {statistics.mean(ts):.3f} s)")

    out = HERE / "compare" / f"vllm_{args.run}_{tag}.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"저장: {out}")


if __name__ == "__main__":      # vLLM 은 엔진을 다른 프로세스로 띄워서 이 줄이 꼭 필요
    main()
