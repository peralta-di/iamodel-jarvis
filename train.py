"""Entrena a Jarvis desde cero sobre tus propios textos (capítulo 5 de "LLMs-from-scratch").

Ejemplos:
    python train.py                                   # usa data/ y valores por defecto
    python train.py --data data/quijote.txt --epochs 3
    python train.py --emb-dim 384 --layers 6 --heads 6 --context-length 256  # modelo más grande
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from jarvis import checkpoint
from jarvis.data import create_dataloader
from jarvis.generate import generate_text
from jarvis.model import GPTConfig, GPTModel
from jarvis.tokenizer import END_OF_TEXT, BPETokenizer


def read_corpus(path: Path) -> str:
    files = sorted(path.rglob("*.txt")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"No se encontraron archivos .txt en {path}")
    print(f"Leyendo {len(files)} archivo(s): {', '.join(f.name for f in files)}")
    # Separamos documentos con el token especial de fin de texto.
    return f"\n{END_OF_TEXT}\n".join(f.read_text(encoding="utf-8") for f in files)


def calc_loss_batch(inputs, targets, model, device):
    inputs, targets = inputs.to(device), targets.to(device)
    logits = model(inputs)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())


@torch.no_grad()
def calc_loss_loader(loader, model, device, num_batches: int | None = None) -> float:
    if len(loader) == 0:
        return float("nan")
    num_batches = min(num_batches or len(loader), len(loader))
    total = 0.0
    for i, (x, y) in enumerate(loader):
        if i >= num_batches:
            break
        total += calc_loss_batch(x, y, model, device).item()
    return total / num_batches


def evaluate(model, train_loader, val_loader, device, eval_iters: int):
    model.eval()
    train_loss = calc_loss_loader(train_loader, model, device, eval_iters)
    val_loss = calc_loss_loader(val_loader, model, device, eval_iters)
    model.train()
    return train_loss, val_loss


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data", help="archivo .txt o carpeta con archivos .txt")
    p.add_argument("--out", default="checkpoints/jarvis", help="carpeta donde guardar el modelo")
    p.add_argument("--vocab-size", type=int, default=2048)
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--emb-dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None, help="detener tras N pasos (opcional)")
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--prompt", default="Jarvis, ", help="texto inicial para las muestras")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # 1. Datos y tokenizador -------------------------------------------------
    text = read_corpus(Path(args.data))
    print(f"Corpus: {len(text):,} caracteres. Entrenando tokenizador BPE "
          f"(vocabulario = {args.vocab_size})...")
    t0 = time.time()
    tokenizer = BPETokenizer.train(text, args.vocab_size)
    token_ids = tokenizer.encode(text)
    print(f"  listo en {time.time() - t0:.1f}s -> {len(token_ids):,} tokens "
          f"(vocabulario real = {tokenizer.vocab_size})")

    split = int(len(token_ids) * (1 - args.val_ratio))
    ctx = args.context_length
    train_loader = create_dataloader(token_ids[:split], args.batch_size, ctx, stride=ctx // 2)
    val_ids = token_ids[split:]
    if len(val_ids) <= ctx:
        raise SystemExit("La parte de validación es muy corta: usa más texto, "
                         "reduce --context-length o aumenta --val-ratio.")
    val_loader = create_dataloader(val_ids, args.batch_size, ctx, stride=ctx,
                                   shuffle=False, drop_last=False)
    if len(train_loader) == 0:
        raise SystemExit("No hay suficientes ejemplos de entrenamiento: reduce --batch-size.")

    # 2. Modelo ---------------------------------------------------------------
    cfg = GPTConfig(vocab_size=tokenizer.vocab_size, context_length=ctx, emb_dim=args.emb_dim,
                    n_heads=args.heads, n_layers=args.layers, drop_rate=args.dropout)
    model = GPTModel(cfg).to(device)
    print(f"Modelo GPT con {model.num_parameters() / 1e6:.2f} M de parámetros")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.max_steps or args.epochs * len(train_loader)
    warmup = max(1, min(100, total_steps // 10))
    # Calentamiento lineal + decaimiento coseno de la tasa de aprendizaje.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warmup if s < warmup
        else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total_steps - warmup)))),
    )

    # 3. Bucle de entrenamiento ---------------------------------------------
    step, best_val = 0, float("inf")
    t0 = time.time()
    model.train()
    for epoch in range(args.epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(x, y, model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % args.eval_every == 0 or step == total_steps:
                train_loss, val_loss = evaluate(model, train_loader, val_loader, device, args.eval_iters)
                print(f"época {epoch + 1} | paso {step:>5}/{total_steps} | "
                      f"pérdida train {train_loss:.3f} | val {val_loss:.3f} | "
                      f"perplejidad val {math.exp(val_loss):.1f} | {time.time() - t0:.0f}s")
                if val_loss < best_val:
                    best_val = val_loss
                    checkpoint.save(args.out, model, tokenizer)
            if step >= total_steps:
                break

        sample = generate_text(model, tokenizer, args.prompt, max_new_tokens=60)
        print(f"\n--- muestra tras época {epoch + 1} ---\n{args.prompt}{sample}\n")
        if step >= total_steps:
            break

    if best_val == float("inf"):  # nunca se evaluó: guardamos el estado final
        checkpoint.save(args.out, model, tokenizer)
    print(f"Modelo guardado en {args.out}/ (mejor pérdida de validación: {best_val:.3f})")
    print(f"Pruébalo con:  python chat.py --model {args.out}")


if __name__ == "__main__":
    main()
