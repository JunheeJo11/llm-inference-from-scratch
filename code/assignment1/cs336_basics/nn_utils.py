import torch
import math
from collections.abc import Callable, Iterable
from typing import Optional

def cross_entropy(logits:torch.Tensor, targets:torch.Tensor)->torch.Tensor:
    """평균 cross-entropy. loss = -log softmax(o)[정답] = logsumexp(o) - o[정답]
    
    logits: (..., vocab_size), targets: (...) 정답 토큰 ID"""

    logits = logits - logits.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(logits).sum(dim=-1))
    correct = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    return (log_sum_exp - correct).mean()

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t+1) * grad
                state["t"] = t + 1

        return loss

# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# opt = SGD([weights], lr=1)
# for t in range(10):
#     opt.zero_grad()
#     loss = (weights**2).mean()
#     print(loss.cpu().item())
#     loss.backward()
#     opt.step()

# # lr tuning
# for lr in [1, 1e1, 1e2, 1e3]:
#     torch.manual_seed(0)
#     weights = torch.nn.Parameter(5*torch.randn((10, 10)))
#     opt = SGD([weights], lr=lr)
#     losses = []
#     for t in range(10):
#         opt.zero_grad()
#         loss = (weights**2).mean()
#         losses.append(loss.item())
#         loss.backward()
#         opt.step()
#     print(f"lr={lr:>6g}: " + " ".join(f"{x:3g}" for x in losses))

def get_lr_cosine_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    
    if it < warmup_iters:  # warmup: 0에서 max까지 직선으로
        return it / warmup_iters * max_learning_rate
    if it <= cosine_cycle_iters:  # cosine: max에서 min까지 반쪽 코사인 곡선으로
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)  # 0 -> 1
        return min_learning_rate + 0.5 * (1 + math.cos(progress * math.pi)) * (max_learning_rate - min_learning_rate)
    return min_learning_rate  # 끝난 뒤: min 유지

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]  # 기울기가 없는 파라미터는 건너뜀
    if not grads:
        return
    # 전체 노름 = sqrt(모든 기울기 원소 제곱의 합)
    total_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)  # 밑줄(_)로 끝나는 함수 = 새 텐서를 만들지 않고 그 자리에서 수정