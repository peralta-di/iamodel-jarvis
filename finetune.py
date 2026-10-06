"""Ajuste fino de Jarvis para seguir instrucciones (capítulo 7 de "LLMs-from-scratch").

Parte de un modelo ya preentrenado con train.py y lo entrena con pares
instrucción → respuesta (data/instrucciones/*.jsonl).

    python finetune.py
    python finetune.py --base checkpoints/jarvis --data mis_instrucciones.jsonl --epochs 5
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jarvis import checkpoint
from jarvis.generate import generate_text
from jarvis.instruct import (IGNORE_INDEX, InstructionDataset, extract_response, format_prompt,
                             load_examples, make_collate_fn)


@torch.no_grad()
def eval_loss(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        if (y == IGNORE_INDEX).all():
            continue
        logits = model(x.to(device))
        total += torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), y.to(device).flatten(), ignore_index=IGNORE_INDEX).item()
        n += 1
    model.train()
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="checkpoints/jarvis", help="modelo preentrenado")
    p.add_argument("--data", default="data/instrucciones", help="archivo .jsonl/.json o carpeta")
    p.add_argument("--out", default="checkpoints/jarvis-chat")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--no-mask-prompt", action="store_true",
                   help="calcular la pérdida también sobre la instrucción")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = checkpoint.load(args.base, device)
    model.train()
    print(f"Modelo base: {args.base} ({model.num_parameters() / 1e6:.2f} M parámetros)")

    data_path = Path(args.data)
    files = sorted(data_path.glob("*.json*")) if data_path.is_dir() else [data_path]
    examples = [ex for f in files for ex in load_examples(f)]
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_ratio))
    val_ex, train_ex = examples[:n_val], examples[n_val:]
    print(f"{len(train_ex)} ejemplos de entrenamiento, {len(val_ex)} de validación")

    collate = make_collate_fn(tokenizer.eot_id, model.cfg.context_length,
                              mask_prompt=not args.no_mask_prompt)
    train_loader = DataLoader(InstructionDataset(train_ex, tokenizer), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(InstructionDataset(val_ex, tokenizer), batch_size=args.batch_size,
                            collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, t0 = float("inf"), time.time()
    print(f"Pérdida de validación inicial: {eval_loss(model, val_loader, device):.3f}")
    for epoch in range(args.epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x.to(device))
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), y.to(device).flatten(), ignore_index=IGNORE_INDEX)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val = eval_loss(model, val_loader, device)
        print(f"época {epoch + 1}/{args.epochs} | pérdida train {loss.item():.3f} | val {val:.3f} "
              f"| perplejidad val {math.exp(min(val, 50)):.1f} | {time.time() - t0:.0f}s")
        if val < best:
            best = val
            checkpoint.save(args.out, model, tokenizer)

    ex = val_ex[0]
    answer = extract_response(generate_text(model, tokenizer, format_prompt(ex["instruction"], ex.get("input", "")),
                                            max_new_tokens=80, temperature=0.0, top_k=None))
    print(f"\nEjemplo de validación\n  Instrucción: {ex['instruction']}\n"
          f"  Esperado:    {ex['output']}\n  Jarvis:      {answer}\n")
    print(f"Modelo ajustado guardado en {args.out}/")
    print(f"Pruébalo con:  python chat.py --model {args.out} --instruct")


if __name__ == "__main__":
    main()
