"""Tokenizador BPE (Byte Pair Encoding) a nivel de bytes, escrito desde cero.

Equivale al capítulo 2 de "LLMs-from-scratch": convertir texto en IDs de tokens.
En lugar de usar `tiktoken` (el tokenizador de GPT-2), entrenamos el nuestro
para que funcione bien con textos en español y sin depender de internet.

Idea de BPE:
  1. Empezamos con los 256 bytes posibles como vocabulario base.
  2. Contamos qué pareja de tokens consecutivos aparece más veces.
  3. La fusionamos en un token nuevo y repetimos hasta llegar al tamaño deseado.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# Separa el texto en "palabras" (con su espacio inicial), números y signos.
# Las fusiones BPE nunca cruzan estos límites.
PRETOKENIZE = re.compile(r" ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+")

END_OF_TEXT = "<|endoftext|>"


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for pair in merges or []:
            self._add_merge(pair)
        self.special_tokens = {END_OF_TEXT: len(self.vocab)}
        self._cache: dict[str, list[int]] = {}

    def _add_merge(self, pair: tuple[int, int]) -> None:
        new_id = 256 + len(self.merges)
        self.merges[pair] = new_id
        self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    @property
    def eot_id(self) -> int:
        return self.special_tokens[END_OF_TEXT]

    # ------------------------------------------------------------------ train
    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        if vocab_size < 257:
            raise ValueError("vocab_size debe ser al menos 257 (256 bytes + 1 especial)")
        tok = cls()
        num_merges = vocab_size - 256 - len(tok.special_tokens)

        # Cada palabra distinta se guarda una sola vez junto a su frecuencia.
        word_counts = Counter(PRETOKENIZE.findall(text.replace(END_OF_TEXT, " ")))
        words = [list(w.encode("utf-8")) for w in word_counts]
        freqs = list(word_counts.values())

        for step in range(num_merges):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for ids, f in zip(words, freqs):
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += f
            if not pair_counts:
                break
            best, count = pair_counts.most_common(1)[0]
            if count < 2:
                break
            tok._add_merge(best)
            new_id = tok.merges[best]
            words = [_merge(ids, best, new_id) for ids in words]
            if verbose and (step + 1) % 100 == 0:
                print(f"  fusión {step + 1}/{num_merges}: {tok.vocab[new_id]!r} ({count} veces)")

        tok.special_tokens = {END_OF_TEXT: len(tok.vocab)}
        return tok

    # ----------------------------------------------------------- encode/decode
    def _encode_word(self, word: str) -> list[int]:
        if word in self._cache:
            return self._cache[word]
        ids = list(word.encode("utf-8"))
        while len(ids) >= 2:
            # Aplica primero la fusión que se aprendió antes (menor rango).
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        self._cache[word] = ids
        return ids

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        ids: list[int] = []
        parts = text.split(END_OF_TEXT) if allow_special else [text]
        for i, part in enumerate(parts):
            if i > 0:
                ids.append(self.eot_id)
            for word in PRETOKENIZE.findall(part):
                ids.extend(self._encode_word(word))
        return ids

    def decode(self, ids: list[int]) -> str:
        inverse_special = {v: k for k, v in self.special_tokens.items()}
        out = bytearray()
        for i in ids:
            if i in inverse_special:
                out.extend(inverse_special[i].encode("utf-8"))
            else:
                out.extend(self.vocab[i])
        return out.decode("utf-8", errors="replace")

    # ------------------------------------------------------------ save / load
    def save(self, path: str | Path) -> None:
        data = {"merges": [list(p) for p in self.merges]}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(p) for p in data["merges"]])


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
