# 11/2 Karpathy GPT: 요소별 손실 변화

## 환경
- RTX 5070 Ti, PyTorch 2.14.1+cu130 (environment.md 참고)
- 데이터: tiny shakespeare, 글자 단위 토크나이저 (vocab 65), train 90% / val 10%
- 손실은 estimate_loss (배치 200개 평균)

## 단계별 결과
| 단계 | 주요 설정 | 최종 train loss | 최종 val loss | 영상 값 (참고) |
|---|---|---|---|---|
| bigram | lr 1e-2, 3000 스텝 | | | 기록 안함 |
| + single head attention | n_embd 32, block 8, lr 1e-3, 5000 스텝 | 기록 안함 | 기록 안함 | 기록 안함 |
| + multi-head (4 head) | | | | 기록 안함 |
| + feedforward | | | | 기록 안함 |
| + residual, block 3개 | | | | 기록 안함 |
| + LayerNorm | | | | 기록 안함 |
| 최종 (scale up) | n_embd 384, 6 layers, 6 head, block 256, dropout 0.2 | | | 약 1.48 |

## 최종 모델
- 파라미터 수: 10.79 M parameters
- 학습 시간 (training time 출력): 8.7 min (520 s), 5000 steps
- 최대 GPU 메모리 (peak GPU memory 출력): 3.43 GB
- 생성 결과 일부:
'''
Boy: meet my heart gracious lord,
But love you blay; I'll fall back out.

BALTHASAR:

RICHARD:
Harry be Uother, Hark! you must do well;
But, I would you sue ob, being, and a thou
Sit king, and I go presently, and to Salisby,
A gib axe out the ladies of a stroke,
Nor striving that a propheth and brow
'''

## 메모
- 새 Ubuntu에 gcc가 없어 Triton 경로(T=1 생성 시 외적 행렬곱)에서 실패, build-essential 설치로 해결
- 평균 트릭 비교에서 allclose가 False: 차이 최댓값 3.2e-8, 0 근처 원소 때문에 기본 atol 1e-8 초과. atol=1e-6으로 판정
- val loss 최저는 step 4500의 1.4832, 이후 4999에서 1.4896으로 소폭 상승 (과적합 신호, train과 val 차이 약 0.39)
- 1080만 파라미터인데 최대 메모리 3.43GB: 가중치보다 활성값과 attention 점수 행렬이 대부분으로 추정, 11/12에 손계산으로 확인 예정