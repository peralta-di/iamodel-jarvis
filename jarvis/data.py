"""Preparación de datos (capítulo 2 de "LLMs-from-scratch").

Usamos una ventana deslizante sobre los IDs de tokens: la entrada son
`max_length` tokens y el objetivo es la misma secuencia desplazada en 1.
Así el modelo aprende a predecir siempre el siguiente token.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class GPTDataset(Dataset):
    def __init__(self, token_ids: list[int], max_length: int, stride: int):
        if len(token_ids) <= max_length:
            raise ValueError(
                f"El texto tiene {len(token_ids)} tokens; se necesitan más de {max_length}. "
                "Usa más texto o reduce --context-length."
            )
        ids = torch.tensor(token_ids, dtype=torch.long)
        starts = range(0, len(ids) - max_length, stride)
        self.inputs = torch.stack([ids[i:i + max_length] for i in starts])
        self.targets = torch.stack([ids[i + 1:i + max_length + 1] for i in starts])

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def create_dataloader(token_ids: list[int], batch_size: int, max_length: int, stride: int,
                      shuffle: bool = True, drop_last: bool = True) -> DataLoader:
    dataset = GPTDataset(token_ids, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
