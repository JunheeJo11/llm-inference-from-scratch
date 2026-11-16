"""3장 Transformer 언어모델 구성 요소.

위치: cs336_basics/model.py
오늘 차례로 이 파일에 모듈을 추가한다.
지금까지: 3.3 Linear, Embedding / 3.4.1 RMSNorm / 3.4.2 SwiGLU / 3.4.3 RoPE
          3.4.4 softmax, scaled dot-product attention / 3.4.5 causal multi-head self-attention
"""
import math

import torch
import torch.nn as nn
from einops import einsum, rearrange


class Linear(nn.Module):
    """bias 없는 선형 변환 y = W x. W 모양은 (out_features, in_features)"""

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()  # nn.Module의 준비 작업을 먼저 실행 (빠뜨리면 파라미터 등록이 안 됨)
        self.in_features = in_features
        self.out_features = out_features

        # W를 (d_out, d_in) 모양으로 만들고 nn.Parameter로 감싸 학습 대상으로 등록
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        # 초기화: 평균 0, 분산 2/(d_in + d_out)인 정규분포를 [-3σ, 3σ]에서 자른 것
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_in), W: (d_out, d_in) -> (..., d_out)
        # 앞쪽 배치 차원(...)이 몇 개든 그대로 두고 d_in 축끼리 곱해서 더한다
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """토큰 ID -> d_model 차원 벡터. 행렬 모양은 (vocab_size, d_model)"""

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        # 초기화: 평균 0, 표준편차 1인 정규분포를 [-3, 3]에서 자른 것
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (...) 정수 텐서 -> (..., d_model)
        # 행렬의 token_ids번째 행들을 골라 온다 (고급 인덱싱)
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """RMSNorm(a_i) = a_i / RMS(a) * g_i,  RMS(a) = sqrt(mean(a^2) + eps)"""

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # 학습되는 gain g. d_model개, 1로 초기화 (처음엔 정규화만 하고 크기는 그대로)
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # 제곱할 때 넘치지 않게 float32로 올려서 계산

        # 마지막 축(d_model) 방향으로 제곱 평균. keepdim=True라 (..., 1) 모양이 되어 x와 나눌 수 있음
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight

        return result.to(in_dtype)  # 원래 dtype으로 되돌려서 반환


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU(x) = x * sigmoid(x)"""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """FFN(x) = W2( SiLU(W1 x) ⊙ W3 x )"""

    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            # 8/3 * d_model을 64의 배수로 올림 (하드웨어 효율)
            d_ff = math.ceil(8 * d_model / 3 / 64) * 64
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # 게이트 쪽 (SiLU를 통과)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # 다시 d_model로 줄이는 쪽
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # 값 쪽 (그대로)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model) -> w1, w3: (..., d_ff) -> 원소별 곱 -> w2: (..., d_model)
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    """RoPE: q, k를 두 칸씩 짝지어 위치 i에 비례한 각도 θ_{i,k} = i / Θ^((2k-2)/d) 만큼 회전"""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        k = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)  # 0, 2, 4, ..., d-2
        inv_freq = 1.0 / (theta ** (k / d_k))  # 짝마다의 회전 속도
        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angle = einsum(pos, inv_freq, "seq, half -> seq half")  # (max_seq_len, d/2)

        # 학습하지 않는 값이라 버퍼로. persistent=False: state_dict에 저장하지 않음
        self.register_buffer("cos", torch.cos(angle), persistent=False)
        self.register_buffer("sin", torch.sin(angle), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k), token_positions: (..., seq_len)
        cos = self.cos[token_positions]  # (..., seq_len, d/2)
        sin = self.sin[token_positions]

        x1 = x[..., 0::2]  # 짝의 첫째 (0, 2, 4, ... 번째 칸)
        x2 = x[..., 1::2]  # 짝의 둘째 (1, 3, 5, ... 번째 칸)

        # 2x2 회전행렬 [cos -sin; sin cos]를 짝 (x1, x2)에 곱한 결과
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        out = torch.stack([out1, out2], dim=-1)  # (..., seq_len, d/2, 2)
        return out.flatten(-2)  # (..., seq_len, d_k), 짝을 원래 자리로


# ---------------------------------------------------------------------------
# 3.4.4 softmax와 scaled dot-product attention
# ---------------------------------------------------------------------------


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """dim 축 방향으로 softmax. 최댓값을 먼저 빼서 exp가 inf로 넘치지 않게 한다"""
    # keepdim=True: 줄어든 축을 크기 1로 남겨 x와 바로 뺄 수 있게 (브로드캐스팅)
    x_max = x.max(dim=dim, keepdim=True).values
    exp = torch.exp(x - x_max)  # 가장 큰 값이 0이 되므로 exp는 최대 1
    return exp / exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Q: (..., queries, d_k), K: (..., keys, d_k), V: (..., keys, d_v)
    mask: (..., queries, keys) bool. True면 본다, False면 안 본다
    반환: (..., queries, d_v)
    """
    d_k = Q.shape[-1]

    # 모든 query와 모든 key의 내적 표. 모양 (..., queries, keys)
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(d_k)

    if mask is not None:
        # False인 칸을 -inf로 채우면 softmax 뒤에 정확히 0이 된다 (exp(-inf) = 0)
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = softmax(scores, dim=-1)  # key 축으로 정규화: 각 query 행의 합이 1

    # 확률 가중치로 V를 섞는다. (..., q, k) x (..., k, d_v) -> (..., q, d_v)
    return einsum(attn, V, "... q k, ... k d -> ... q d")


# ---------------------------------------------------------------------------
# 3.4.5 causal multi-head self-attention
# ---------------------------------------------------------------------------


class MultiHeadSelfAttention(nn.Module):
    """causal multi-head self-attention. d_k = d_v = d_model / num_heads

    rope를 넘기면 q, k에 RoPE를 적용하고, None이면 위치 정보 없이 계산한다.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model은 num_heads로 나누어떨어져야 함"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rope = rope

        # 모든 head의 W_Q를 세로로 쌓은 행렬 하나 (h * d_k, d_model) = (d_model, d_model)
        # 이름은 테스트 state_dict 키(q_proj, k_proj, v_proj, output_proj)에 맞춘다
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: (..., seq_len, d_model)
        seq_len = x.shape[-2]

        # 1) 행렬 곱 세 번으로 모든 head의 q, k, v를 한꺼번에 만든다
        q = self.q_proj(x)  # (..., seq, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2) d_model 축을 (head, d_head)로 쪼개고 head를 배치 축처럼 앞으로 뺀다
        q = rearrange(q, "... seq (h d) -> ... h seq d", h=self.num_heads)
        k = rearrange(k, "... seq (h d) -> ... h seq d", h=self.num_heads)
        v = rearrange(v, "... seq (h d) -> ... h seq d", h=self.num_heads)

        # 3) RoPE는 q와 k에만. head 축은 배치처럼 취급되므로 모든 head에 같은 회전이 들어간다
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)  # 0, 1, ..., seq_len-1
            # (..., seq) -> (..., 1, seq): head 축 자리에 1을 끼워 넣어 모든 head에 브로드캐스팅
            pos = token_positions.unsqueeze(-2)
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # 4) causal mask: query i는 key j <= i 만 본다. 아래 삼각형(대각선 포함)이 True
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(q, k, v, mask)  # (..., h, seq, d_head)

        # 5) head들을 다시 이어 붙여(Concat) d_model로 되돌리고 출력 projection
        out = rearrange(out, "... h seq d -> ... seq (h d)")
        return self.output_proj(out)

# -----------------------------------------------------------------------------------------------
# 3.5 Transformer block과 언어모델 전체
# -----------------------------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """pre-norm transformer block
    
    y = x + MHA(RMSNorm(x))         (식 15, 첫째 서브층)
    z = y + FFN(RMSNorm(y))         (둘째 서브층)
    """

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            rope: RotaryPositionalEmbedding | None=None,
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    """토큰 ID -> 임베딩 -> block x num layers -> 최종 RMSNorm -> LM head -> logits"""

    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float = 10000.0,
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # RoPE는 학습 파라미터가 없으니 하나만 만들어 모든 층이 같이 씀
        rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length, device=device)

        # nn.ModuleList: 모듈 목록
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, rope=rope, device=device, dtype=dtype) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        # in_indices: (batch, seq) 정수.
        seq_len = in_indices.shape[-1]
        assert seq_len <= self.context_length, "입력이 context_length보다 길다"
        token_positions = torch.arange(seq_len, device=in_indices.device)

        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)