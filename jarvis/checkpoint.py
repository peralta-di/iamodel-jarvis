"""Guardar y cargar el modelo entrenado junto con su tokenizador."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import GPTConfig, GPTModel
from .tokenizer import BPETokenizer


def save(out_dir: str | Path, model: GPTModel, tokenizer: BPETokenizer) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.cfg.to_dict(), "state_dict": model.state_dict()},
               out_dir / "model.pt")
    tokenizer.save(out_dir / "tokenizer.json")


def load(out_dir: str | Path, device: str = "cpu") -> tuple[GPTModel, BPETokenizer]:
    out_dir = Path(out_dir)
    ckpt = torch.load(out_dir / "model.pt", map_location=device, weights_only=True)
    model = GPTModel(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, BPETokenizer.load(out_dir / "tokenizer.json")
