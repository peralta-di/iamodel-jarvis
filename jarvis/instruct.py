"""Ajuste fino con instrucciones (capítulo 7 de "LLMs-from-scratch").

Cada ejemplo es un par instrucción → respuesta que se escribe con una plantilla
fija (estilo Alpaca). El modelo aprende a continuar la plantilla escribiendo la
respuesta, y así pasa de "continuar texto" a "seguir instrucciones".
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import BPETokenizer

IGNORE_INDEX = -100  # cross_entropy ignora estas posiciones


def format_prompt(instruction: str, input_text: str = "") -> str:
    prompt = ("A continuación hay una instrucción que describe una tarea. "
              "Escribe una respuesta que la complete adecuadamente.\n\n"
              f"### Instrucción:\n{instruction.strip()}")
    if input_text.strip():
        prompt += f"\n\n### Entrada:\n{input_text.strip()}"
    return prompt + "\n\n### Respuesta:\n"


def load_examples(path: str | Path) -> list[dict]:
    """Lee un archivo .jsonl (un objeto por línea) o .json (una lista)."""
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix == ".jsonl":
        examples = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        examples = json.loads(text)
    for ex in examples:
        if "instruction" not in ex or "output" not in ex:
            raise ValueError(f"Cada ejemplo necesita 'instruction' y 'output': {ex}")
    return examples


class InstructionDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer: BPETokenizer):
        self.items = []
        for ex in examples:
            prompt_ids = tokenizer.encode(format_prompt(ex["instruction"], ex.get("input", "")))
            answer_ids = tokenizer.encode(ex["output"]) + [tokenizer.eot_id]
            self.items.append((prompt_ids, answer_ids))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.items[idx]


def make_collate_fn(pad_id: int, max_length: int, mask_prompt: bool = True):
    """Rellena cada lote a la misma longitud y construye los objetivos.

    - Los objetivos son la entrada desplazada una posición.
    - El relleno se marca con IGNORE_INDEX para que no cuente en la pérdida.
    - Con `mask_prompt`, tampoco cuenta la instrucción: solo aprendemos la respuesta.
    """
    def collate(batch):
        seqs = []
        for prompt_ids, answer_ids in batch:
            # Si no cabe todo, la respuesta conserva al menos la mitad del contexto
            # y la instrucción se recorta por la izquierda (se queda su final).
            answer_ids = answer_ids[:max(1, (max_length + 1) // 2)] \
                if len(prompt_ids) + len(answer_ids) > max_length + 1 else answer_ids
            prompt_ids = prompt_ids[-(max_length + 1 - len(answer_ids)):]
            seqs.append((prompt_ids + answer_ids, len(prompt_ids)))
        longest = max(len(ids) for ids, _ in seqs)
        inputs, targets = [], []
        for ids, prompt_len in seqs:
            padded = ids + [pad_id] * (longest - len(ids))
            x = torch.tensor(padded[:-1])
            y = torch.tensor(padded[1:])
            y[len(ids) - 1:] = IGNORE_INDEX
            if mask_prompt:
                y[:max(0, prompt_len - 1)] = IGNORE_INDEX
            inputs.append(x)
            targets.append(y)
        return torch.stack(inputs), torch.stack(targets)

    return collate


def extract_response(generated: str) -> str:
    """Quita lo que el modelo pueda añadir tras la respuesta (otra plantilla, etc.)."""
    return generated.split("### Instrucción:")[0].split("<|endoftext|>")[0].strip()
