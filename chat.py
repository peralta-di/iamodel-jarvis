"""Habla con Jarvis: carga el modelo entrenado y genera texto a partir de lo que escribas.

    python chat.py                                  # modo interactivo
    python chat.py --prompt "Jarvis, ¿qué es"       # una sola respuesta
"""

from __future__ import annotations

import argparse

import torch

from jarvis import checkpoint
from jarvis.generate import generate_text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="checkpoints/jarvis")
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8,
                   help="0 = siempre la palabra más probable; más alto = más creativo")
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = checkpoint.load(args.model, device)

    def reply(prompt: str) -> str:
        return generate_text(model, tokenizer, prompt, args.max_new_tokens,
                             args.temperature, args.top_k or None)

    if args.prompt is not None:
        print(args.prompt + reply(args.prompt))
        return

    print("Jarvis listo. Escribe el inicio de un texto y Jarvis lo continuará (Ctrl+C para salir).")
    try:
        while True:
            prompt = input("\nTú: ").strip()
            if prompt.lower() in {"salir", "exit", "quit"}:
                break
            if prompt:
                print(f"Jarvis: {prompt}{reply(prompt)}")
    except (KeyboardInterrupt, EOFError):
        pass
    print("\nHasta luego.")


if __name__ == "__main__":
    main()
