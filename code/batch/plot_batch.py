"""bench_batch 결과 JSON에서 그림만 다시 그리기 (측정은 다시 안 함)

위치: assignment1-basics/scratch/plot_batch.py
실행: uv run python scratch/plot_batch.py            (runs/batch/ 의 가장 최근 결과)
      uv run python scratch/plot_batch.py runs/batch/batch_<시각>.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))   # 같은 폴더의 bench_batch.py를 불러오려고
from bench_batch import plot                     # noqa: E402  그림 함수는 bench_batch.py 것을 그대로 씀

if len(sys.argv) > 1:
    path = Path(sys.argv[1])
else:
    cands = sorted(p for p in Path("runs/batch").glob("*.json") if not p.stem.endswith("_theory"))
    path = cands[-1]

data = json.loads(path.read_text())
# JSON에 저장하면 딕셔너리 키가 문자열이 된다 ({16: ...} -> {"16": ...}). 그림 함수는 숫자 키로 찾으므로 되돌림
for r in data["curve"]:
    for k in ("ttft_ms", "ttft_ms_std", "prefill_tok_s"):
        r[k] = {int(P): v for P, v in r[k].items()}

out = path.with_suffix(".png")
plot(data["curve"], data["workload"], data["config"], out)
print(f"저장: {out}")
