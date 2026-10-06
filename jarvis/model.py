"""Arquitectura GPT escrita desde cero (capítulos 3 y 4 de "LLMs-from-scratch").

Piezas:
  - MultiHeadAttention: atención causal con varias cabezas (cap. 3).
  - LayerNorm, GELU, FeedForward: bloques básicos (cap. 4).
  - TransformerBlock: atención + red feed-forward con conexiones residuales.
  - GPTModel: embeddings de tokens y posiciones + N bloques + capa de salida.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int = 2048
    context_length: int = 128   # tokens máximos que el modelo "ve" a la vez
    emb_dim: int = 256          # tamaño de cada vector de embedding
    n_heads: int = 4            # cabezas de atención
    n_layers: int = 4           # bloques transformer
    drop_rate: float = 0.1
    qkv_bias: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float,
                 num_heads: int, qkv_bias: bool = False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out debe ser divisible por num_heads"
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # Máscara causal: cada token solo puede mirar a los anteriores.
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n_tokens, _ = x.shape

        # (b, n_tokens, d_out) -> (b, num_heads, n_tokens, head_dim)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.W_query(x))
        k = split_heads(self.W_key(x))
        v = split_heads(self.W_value(x))

        # Atención de producto escalar escalada.
        scores = q @ k.transpose(2, 3) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:n_tokens, :n_tokens], float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))

        context = (weights @ v).transpose(1, 2).contiguous().view(b, n_tokens, -1)
        return self.out_proj(context)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim, d_out=cfg.emb_dim, context_length=cfg.context_length,
            dropout=cfg.drop_rate, num_heads=cfg.n_heads, qkv_bias=cfg.qkv_bias,
        )
        self.ff = FeedForward(cfg.emb_dim)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_shortcut(self.att(self.norm1(x)))  # conexión residual 1
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))   # conexión residual 2
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = in_idx.shape
        pos = torch.arange(seq_len, device=in_idx.device)
        x = self.drop_emb(self.tok_emb(in_idx) + self.pos_emb(pos))
        x = self.trf_blocks(x)
        return self.out_head(self.final_norm(x))  # logits: (batch, seq_len, vocab_size)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
