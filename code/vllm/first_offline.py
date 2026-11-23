"""vLLM 첫 실행: 오프라인 추론 + KV 캐시 크기 손계산과 로그 대조

위치: ~/llm-study/vllm-lab/first_offline.py
실행 (vllm-lab 폴더에서, 가상환경을 켠 상태):
    python first_offline.py 2>&1 | tee logs/first_offline.txt

- 오프라인 추론: 서버 없이 파이썬 안에서 LLM 객체를 만들어 바로 generate
- vLLM은 시작할 때 GPU 메모리의 gpu_memory_utilization 비율만큼을 미리 잡고,
  모델 가중치를 올린 뒤 남는 공간을 전부 KV 캐시 블록으로 나눠 둔다 (PagedAttention)
"""
import time

from transformers import AutoConfig
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
GPU_MEM_UTIL = 0.6     # GPU 메모리의 60%만 쓰기 (윈도우 화면과 다른 앱 몫을 남김)
MAX_MODEL_LEN = 4096   # 요청 하나가 쓸 수 있는 최대 토큰 수 (프롬프트 + 생성)


def kv_bytes_per_token(model: str, dtype_bytes: int = 2) -> int:
    """KV 캐시 손계산: 토큰 하나당 = 2(K와 V) x 층 수 x KV 헤드 수 x 헤드 차원 x 바이트"""
    cfg = AutoConfig.from_pretrained(model)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    print(f"[config] 층 {cfg.num_hidden_layers}, hidden {cfg.hidden_size}, query 헤드 {cfg.num_attention_heads}, "
          f"KV 헤드 {n_kv}, 헤드 차원 {head_dim}")
    per_token = 2 * cfg.num_hidden_layers * n_kv * head_dim * dtype_bytes
    if n_kv != cfg.num_attention_heads:
        full = 2 * cfg.num_hidden_layers * cfg.num_attention_heads * head_dim * dtype_bytes
        print(f"[손계산] GQA 덕분에 KV 캐시가 {full / per_token:.0f}배 작음 (KV 헤드를 query 헤드 수만큼 두면 토큰당 {full:,}바이트)")
    return per_token


def main():
    per_token = kv_bytes_per_token(MODEL)
    print(f"[손계산] KV 캐시 토큰당 {per_token:,}바이트 ({per_token / 1024:.1f} KiB, bf16 기준)")

    t0 = time.perf_counter()
    llm = LLM(model=MODEL, gpu_memory_utilization=GPU_MEM_UTIL, max_model_len=MAX_MODEL_LEN, seed=0)
    print(f"\n[시작] LLM 객체 만들기까지 {time.perf_counter() - t0:.1f}초 (가중치 로드, 컴파일, 메모리 확보 포함)")

    prompts = [
        "Once upon a time,",
        "The capital of South Korea is",
        "Explain in one sentence why the KV cache speeds up LLM inference:",
        "def fibonacci(n):",
    ]
    params = SamplingParams(temperature=0.0, max_tokens=64)   # greedy: 매번 같은 결과

    # 1) 한 번 돌려 보기
    outputs = llm.generate(prompts, params)
    print("\n[생성 결과] greedy, 최대 64토큰")
    for out in outputs:
        text = out.outputs[0].text.replace("\n", " ")
        print(f"  {out.prompt!r} -> {text[:120]!r}")

    # 2) 간단한 속도 감각 (워밍업 1회 뒤 3회)
    for batch in (1, 4, 16):
        ps = (prompts * 4)[:batch]
        llm.generate(ps, params, use_tqdm=False)          # 워밍업
        times, n_tok = [], 0
        for _ in range(3):
            t = time.perf_counter()
            outs = llm.generate(ps, params, use_tqdm=False)
            times.append(time.perf_counter() - t)
            n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
        avg = sum(times) / len(times)
        print(f"[속도] 요청 {batch:2d}개 동시: {avg * 1000:7.1f} ms, 생성 토큰 {n_tok}개, {n_tok / avg:8.1f} tok/s")

    print("\n로그에서 찾을 줄: 'KV cache' 또는 'Maximum concurrency'가 들어간 줄 (KV 캐시 토큰 수, 동시 요청 수)")
    print(f"그 토큰 수 x {per_token:,}바이트 = vLLM이 잡은 KV 캐시 크기. 이 값을 로그의 GiB와 비교")


if __name__ == "__main__":
    main()