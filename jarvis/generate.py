"""Generación de texto (capítulo 5 de "LLMs-from-scratch").

El modelo predice una distribución sobre el siguiente token; elegimos uno,
lo añadimos a la secuencia y repetimos. `temperature` y `top_k` controlan
qué tan "creativo" (aleatorio) es el resultado.
"""

from __future__ import annotations

import torch

from .model import GPTModel
from .tokenizer import BPETokenizer


@torch.no_grad()
def generate(model: GPTModel, idx: torch.Tensor, max_new_tokens: int,
             temperature: float = 0.0, top_k: int | None = None,
             eos_id: int | None = None) -> torch.Tensor:
    context_size = model.cfg.context_length
    for _ in range(max_new_tokens):
        logits = model(idx[:, -context_size:])[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < top_logits[:, -1:], float("-inf"))

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:  # decodificación codiciosa (greedy)
            next_id = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (next_id == eos_id).all():
            break
        idx = torch.cat([idx, next_id], dim=1)
    return idx


def generate_text(model: GPTModel, tokenizer: BPETokenizer, prompt: str,
                  max_new_tokens: int = 100, temperature: float = 0.8,
                  top_k: int | None = 40) -> str:
    """Devuelve solo el texto nuevo generado a continuación de `prompt`."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    ids = tokenizer.encode(prompt) or [tokenizer.eot_id]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(model, idx, max_new_tokens, temperature, top_k, eos_id=tokenizer.eot_id)
    model.train(was_training)
    return tokenizer.decode(out[0, len(ids):].tolist())
