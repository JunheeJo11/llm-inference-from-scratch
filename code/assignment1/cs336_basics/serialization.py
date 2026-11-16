# 5.2 Checkpointing
import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """학습을 이어서 하는 데 필요한 세 가지(모델 가중치, 옵티마이저 상태, 스텝 번호)를 한 파일에 저장"""
    checkpoint = {
        "model": model.state_dict(),          # 이름 -> 가중치 텐서 사전
        "optimizer": optimizer.state_dict(),  # AdamW의 m, v, t와 lr, betas 같은 설정
        "iteration": iteration,               # 몇 스텝까지 했는지 (학습률 스케줄을 이어 가려면 필요)
    }
    torch.save(checkpoint, out)  # out은 파일 경로든 열린 파일 객체든 됨


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """저장해 둔 상태를 model과 optimizer에 그대로 되돌리고, 저장했던 스텝 번호를 돌려준다"""
    # map_location="cpu": GPU에서 저장한 파일도 일단 CPU로 읽음.
    # load_state_dict가 각 값을 model이 있는 장치(GPU면 GPU)로 다시 복사해 준다
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]