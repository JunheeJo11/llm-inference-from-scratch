#!/usr/bin/env bash
# 11/27 다시 재기: prefix caching 끄고 (내 구현과 같은 조건) bf16 6개 설정
set -u
cd "$(dirname "$0")"
mkdir -p logs/1127

python compare_same_cond.py --slots 4 --grid --no-prefix-cache          2>&1 | tee logs/1127/s4_grid_nopc.txt
python compare_same_cond.py --slots 4 --grid --eager --no-prefix-cache  2>&1 | tee logs/1127/s4_grid_eager_nopc.txt
for s in 16 64; do
  python compare_same_cond.py --slots $s --no-prefix-cache          2>&1 | tee logs/1127/s${s}_nopc.txt
  python compare_same_cond.py --slots $s --eager --no-prefix-cache  2>&1 | tee logs/1127/s${s}_eager_nopc.txt
done

echo "모두 끝남: $(date)"
