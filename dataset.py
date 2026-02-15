from typing import Tuple
from torch.utils.data import Dataset
import numpy as np
import torch
from torch import Tensor

class TokenDataset(Dataset):
    def __init__(self, path: str, block_size: int = 1024, stride: int | None = None):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.stride = stride or block_size // 2
        offset = np.random.randint(0, self.stride)
        self.start_pos = np.arange(offset, len(self.data) - block_size, self.stride)
        # np.random.shuffle(self.start_pos)

    def __len__(self):
        return len(self.start_pos)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        start = self.start_pos[idx]
        x = torch.from_numpy(
            self.data[start:start+self.block_size].astype(np.int64)
        )
        y = torch.from_numpy(
            self.data[start+1:start+1+self.block_size].astype(np.int64)
        )
        return x, y
