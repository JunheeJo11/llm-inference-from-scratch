# 2026-11-23 vLLM 첫 실행

## 환경
- vllm 0.34.0, torch 2.14.0+cu132 (CUDA 13.2), RTX 5070 Ti (12, 0), Python 3.12
- Qwen/Qwen2.5-0.5B-Instruct, bf16, gpu_memory_utilization 0.6 (9.55 GiB), max_model_len 4096
- WSL: Pinned host memory 미지원 경고

## 오프라인 (first_offline.py)
- 가중치 0.93 GiB, KV 캐시 10,432블록 x 64토큰 = 667,648토큰 (7.64 GiB), 최대 동시 163개
- KV 캐시 토큰당: 손계산 12,288바이트 (24층 x KV 헤드 2 x 헤드 차원 64 x K,V x 2바이트), 로그 역산 약 12,287바이트
- GQA (query 14 : KV 2)로 KV 캐시 7배 절약
- 시작 142.6초 (엔진 초기화 107.9초, 컴파일 7.3초)
- greedy 64토큰 (워밍업 1, 3회 평균): 동시 1개 138.3 ms 463 tok/s / 4개 146.9 ms 1,742 tok/s / 16개 147.5 ms 6,941 tok/s

## 서버 (vllm serve, curl)
- completions: prompt "The capital of South Korea is", max_tokens 32, temperature 0
  -> 엉뚱한 객관식, 프롬프트 6토큰, 생성 32토큰, finish_reason length (curl_completions.json)
- chat/completions: user "What is the capital of South Korea?", max_tokens 32, temperature 0
  -> "The capital of South Korea is Seoul.", 프롬프트 37토큰 (chat template), 생성 9토큰, finish_reason stop (curl_chat.json)
- 오프라인과 completions 출력 동일 (greedy 재현성)

## 해석
- 요청 16배에 시간 +7%: decode가 memory-bound라 가중치를 한 스텝에 한 번 읽어 여러 요청에 같이 씀
- 같은 모델, 같은 질문인데 이어 쓰기 형식과 대화 형식에서 정답 여부가 갈림