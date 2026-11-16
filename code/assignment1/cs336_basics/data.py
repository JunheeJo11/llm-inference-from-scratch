# 5.1 Data Loader

import numpy as np
import numpy.typing as npt
import torch

def get_batch(
        dataset: npt.NDArray, batch_size: int, context_length: int, device:str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(dataset)
    starts = np.random.randint(0, n-context_length, size=batch_size)
    offsets = np.arange(context_length+1)
    idx = starts[:, None] + offsets[None, :]
    chunk = torch.from_numpy(dataset[idx].astype(np.int64))
    x = chunk[:, :-1]
    y = chunk[:, 1:]
    return x.to(device), y.to(device)
