"""11/27 같은 조건 비교 1단계: 내 모델을 vLLM이 읽는 모양으로 내보내고, 내 구현 쪽 기준 숫자를 잰다

위치: ~/llm-study/assignment1-basics/scratch/export_for_vllm.py
실행: uv run python -u scratch/export_for_vllm.py --run ts_base

하는 일 (과제 환경, torch 2.11 + cu130)
  [1] runs/<run>/config.json 과 ckpt_final.pt 로 내 TransformerLM 을 불러온다
  [2] 같은 조건 비교에 쓸 요청 256개를 만든다 (seed 고정, P 8~128, n 16~128, 11/26 실험 B와 같은 분포)
  [3] 앞 32개 요청의 정답 토큰을 내 KV 캐시 코드(generate_with_cache, greedy, fp32)로 만든다
  [4] 같은 256개를 내 continuous batching(run_continuous)으로 돌려 처리량을 잰다 (슬롯 4, 16, 64, fp32와 bf16)
  [5] 가중치를 HuggingFace Llama 이름과 모양으로 바꿔 vllm-lab/models/<run>_hf/ 에 저장한다

내 모델이 Llama 형식으로 바뀌는 이유
  CS336 과제1 구조(앞 RMSNorm, SwiGLU, RoPE, bias 없음, 임베딩과 출력층 따로)가 Llama와 같다.
  다른 점은 RoPE가 짝을 묶는 방식 하나뿐이다.
    내 RoPE   : (0,1) (2,3) (4,5) ... 붙어 있는 두 칸을 한 짝으로 회전
    HF Llama  : (0,16) (1,17) ... 앞 절반과 뒤 절반을 한 짝으로 회전 (head_dim 32일 때)
  그래서 q_proj, k_proj 의 출력 행 순서를 헤드마다 [0,2,4,...,30, 1,3,...,31] 로 바꿔 준다.
  q와 k에 같은 순서 바꿈을 하면 q와 k의 내적은 그대로라 attention 결과가 같다.
"""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.kv_cache import generate_with_cache
import cs336_basics.batch_infer as bi

HOME = Path.home()
EOT = b"<|endoftext|>"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(run_dir: Path, device: str, dtype):
    """11/26 bench_batch.py 의 load_model 과 같은 방식"""
    cfg = json.loads((run_dir / "config.json").read_text())
    model = TransformerLM(
        cfg["vocab_size"], cfg["context_length"], cfg["d_model"], cfg["num_layers"],
        cfg["num_heads"], cfg["d_ff"], cfg["rope_theta"], device=device,
    )
    ckpt = torch.load(run_dir / "ckpt_final.pt", map_location="cpu")
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}  # torch.compile 접두어 제거
    model.load_state_dict(sd)
    model.to(dtype)
    model.eval()
    return model, cfg, sd


def find_eot_id():
    """토크나이저 vocab.pkl 에서 <|endoftext|> 의 번호를 찾는다 (vocab 은 번호 -> bytes 사전)"""
    import pickle
    p = Path("out/ts_bpe/vocab.pkl")
    if not p.exists():
        return None
    vocab = pickle.loads(p.read_bytes())
    for i, b in vocab.items():
        if b == EOT:
            return int(i)
    return None


def token_pool():
    """프롬프트를 잘라 올 TinyStories 검증 데이터 토큰 (11/16 학습 때 만든 .npy)"""
    cands = sorted(Path("data").glob("*valid*.npy")) or sorted(Path("data").glob("*.npy"))
    if not cands:
        raise SystemExit("data/ 에 .npy 토큰 파일이 없어요. ls data/ 결과를 붙여 주세요")
    arr = np.load(cands[0], mmap_mode="r")
    print(f"프롬프트 출처: {cands[0]} ({len(arr):,}토큰)")
    return arr


def make_requests(arr, n_req, seed):
    rng = random.Random(seed)
    reqs = []
    for _ in range(n_req):
        P, n = rng.randint(8, 128), rng.randint(16, 128)
        s = rng.randrange(0, len(arr) - P)
        reqs.append({"prompt": [int(t) for t in arr[s:s + P]], "n": n})
    return reqs


# ------------------------------------------------------------------ 가중치 이름 바꾸기
def to_hf(sd, cfg):
    H, d = cfg["num_heads"], cfg["d_model"]
    dk = d // H
    perm = torch.cat([torch.arange(0, dk, 2), torch.arange(1, dk, 2)])   # [0,2,...,30, 1,3,...,31]

    def rope_perm(w):                       # w: (H*dk, d) -> 헤드마다 출력 행 순서 바꾸기
        return w.reshape(H, dk, -1)[:, perm, :].reshape(H * dk, -1)

    out, used = {}, set()

    def put(src, dst, f=None):
        w = sd[src].float()
        out[dst] = (f(w) if f else w).contiguous().clone()
        used.add(src)

    put("token_embeddings.weight", "model.embed_tokens.weight")
    for i in range(cfg["num_layers"]):
        a, m = f"layers.{i}.", f"model.layers.{i}."
        put(a + "ln1.weight", m + "input_layernorm.weight")
        put(a + "ln2.weight", m + "post_attention_layernorm.weight")
        put(a + "attn.q_proj.weight", m + "self_attn.q_proj.weight", rope_perm)
        put(a + "attn.k_proj.weight", m + "self_attn.k_proj.weight", rope_perm)
        put(a + "attn.v_proj.weight", m + "self_attn.v_proj.weight")
        put(a + "attn.output_proj.weight", m + "self_attn.o_proj.weight")
        put(a + "ffn.w1.weight", m + "mlp.gate_proj.weight")   # SwiGLU: w2(silu(w1 x) * w3 x)
        put(a + "ffn.w3.weight", m + "mlp.up_proj.weight")
        put(a + "ffn.w2.weight", m + "mlp.down_proj.weight")
    put("ln_final.weight", "model.norm.weight")
    put("lm_head.weight", "lm_head.weight")

    left = [k for k in sd if k not in used and "rope" not in k.lower()]
    if left:
        raise SystemExit(f"이름을 못 바꾼 가중치가 있어요: {left}\n이 줄을 붙여 주세요")
    return out


def hf_config(cfg, eot):
    d, H = cfg["d_model"], cfg["num_heads"]
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": cfg["vocab_size"],
        "hidden_size": d,
        "intermediate_size": cfg["d_ff"],
        "num_hidden_layers": cfg["num_layers"],
        "num_attention_heads": H,
        "num_key_value_heads": H,              # GQA 없음 (헤드마다 자기 key, value)
        "head_dim": d // H,
        "hidden_act": "silu",
        "max_position_embeddings": cfg["context_length"],
        "rms_norm_eps": 1e-5,                  # 내 RMSNorm 기본값 (다르면 고칠 것)
        "rope_theta": float(cfg["rope_theta"]),
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
        "torch_dtype": "float32",
        "bos_token_id": eot,
        "eos_token_id": eot,
    }


# ------------------------------------------------------------------ 내 구현 처리량
def bench_mine(model, reqs, slots_list, warmup, repeats):
    prompts = [r["prompt"] for r in reqs]
    max_new = [r["n"] for r in reqs]
    total = sum(max_new)
    res = {}
    for S in slots_list:
        for _ in range(warmup):
            bi.run_continuous(model, prompts[:32], max_new[:32], S)
        ts = []
        for _ in range(repeats):
            sync()
            t0 = time.perf_counter()
            bi.run_continuous(model, prompts, max_new, S)
            sync()
            ts.append(time.perf_counter() - t0)
        tps = [total / t for t in ts]
        res[S] = {"sec": ts, "tok_s_mean": statistics.mean(tps),
                  "tok_s_std": statistics.stdev(tps) if len(tps) > 1 else 0.0}
        print(f"  슬롯 {S:>3}: {res[S]['tok_s_mean']:>9,.0f} ± {res[S]['tok_s_std']:,.0f} tok/s "
              f"(256개, {total:,}토큰, {statistics.mean(ts):.2f} s)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ts_base")
    ap.add_argument("--n-req", type=int, default=256)
    ap.add_argument("--n-ref", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1127)
    ap.add_argument("--slots", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-bench", action="store_true", help="내 구현 처리량 측정 건너뛰기")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run_dir = Path("runs") / args.run
    out_dir = HOME / "llm-study/vllm-lab/models" / f"{args.run}_hf"
    cmp_dir = HOME / "llm-study/vllm-lab/compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmp_dir.mkdir(parents=True, exist_ok=True)

    # [1] 모델
    model, cfg, sd = load_model(run_dir, "cuda", torch.float32)
    eot = find_eot_id()
    print(f"[1] {args.run}: {cfg}\n    <|endoftext|> 번호 {eot}, torch {torch.__version__}")

    # [2] 요청
    reqs = make_requests(token_pool(), args.n_req, args.seed)
    tot = sum(r["n"] for r in reqs)
    print(f"[2] 요청 {len(reqs)}개 (seed {args.seed}), 생성 토큰 합 {tot:,}")

    # [3] 정답 토큰 (fp32, greedy, 단독 생성)
    with torch.no_grad():
        for r in reqs[:args.n_ref]:
            r["ref"] = [int(t) for t in generate_with_cache(model, r["prompt"], r["n"], temperature=0)]
    print(f"[3] 앞 {args.n_ref}개 요청의 정답 토큰 생성 (generate_with_cache, fp32, greedy)")

    # [5] 내보내기 (측정보다 먼저 해 두면 측정 중에 vLLM 쪽 준비를 볼 수 있음)
    hf_sd = to_hf(sd, cfg)
    (out_dir / "config.json").write_text(json.dumps(hf_config(cfg, eot), indent=2))
    try:
        from safetensors.torch import save_file
        save_file(hf_sd, str(out_dir / "model.safetensors"))
        wfile = "model.safetensors"
    except ImportError:
        torch.save(hf_sd, out_dir / "pytorch_model.bin")   # vLLM 쪽 스크립트가 safetensors 로 바꿔 줌
        wfile = "pytorch_model.bin"
    req_path = cmp_dir / f"requests_{args.run}.json"
    req_path.write_text(json.dumps({"run": args.run, "seed": args.seed, "eot": eot,
                                    "context_length": cfg["context_length"], "requests": reqs}))
    n_par = sum(v.numel() for v in hf_sd.values())
    print(f"[5] 저장: {out_dir}/config.json, {wfile} (파라미터 {n_par:,}개)\n    요청: {req_path}")

    # [4] 내 구현 처리량
    if args.skip_bench:
        return
    result = {"run": args.run, "torch": torch.__version__, "seed": args.seed, "tokens": tot,
              "warmup": args.warmup, "repeats": args.repeats, "tf32": False}
    for name, dt in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
        print(f"[4] 내 continuous batching {name}")
        try:
            m, _, _ = load_model(run_dir, "cuda", dt)
            with torch.no_grad():
                result[name] = bench_mine(m, reqs, args.slots, args.warmup, args.repeats)
            del m
            torch.cuda.empty_cache()
        except Exception as e:                       # bf16 에서 캐시 dtype 이 안 맞는 경우 등
            print(f"  {name} 실패: {type(e).__name__}: {e}")
            result[name] = {"error": repr(e)}
    p = cmp_dir / f"mine_{args.run}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n저장: {p}")


if __name__ == "__main__":
    main()
