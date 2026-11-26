#!/usr/bin/env bash
# 11/27 vLLM 측정 묶음 (vllm-lab 폴더에서 venv를 켠 뒤 실행)
# 설정마다 엔진을 새로 띄워야 해서(슬롯 수와 eager는 시작할 때 정해짐) 프로세스를 따로 돌린다.
# 한 줄이 끝나야 다음 줄이 시작되므로 GPU를 두 프로세스가 동시에 쓰지 않는다.
set -u
cd "$(dirname "$0")"
mkdir -p logs/1127

# 1) 배치 1 격자 + 슬롯 4 처리량: CUDA graph 켬 / 끔
python compare_same_cond.py --slots 4 --grid          2>&1 | tee logs/1127/s4_grid.txt
python compare_same_cond.py --slots 4 --grid --eager  2>&1 | tee logs/1127/s4_grid_eager.txt

# 2) 슬롯 16, 64 처리량: 켬 / 끔
for s in 16 64; do
  python compare_same_cond.py --slots $s          2>&1 | tee logs/1127/s${s}.txt
  python compare_same_cond.py --slots $s --eager  2>&1 | tee logs/1127/s${s}_eager.txt
done

echo "모두 끝남: $(date)"
