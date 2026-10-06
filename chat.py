"""Habla con Jarvis: carga el modelo entrenado y genera texto a partir de lo que escribas.

    python chat.py                                  # modo interactivo
    python chat.py --prompt "Jarvis, ¿qué es"       # una sola respuesta
    python chat.py --model checkpoints/jarvis-chat --instruct   # modelo ajustado con finetune.py
"""

from __future__ import annotations

import argparse

import torch

from jarvis import checkpoint
from jarvis.generate import generate_text
from jarvis.instruct import extract_response, format_prompt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="checkpoints/jarvis")
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8,
                   help="0 = siempre la palabra más probable; más alto = más creativo")
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--instruct", action="store_true",
                   help="tratar lo que escribes como una instrucción (modelo de finetune.py)")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = checkpoint.load(args.model, device)

    def reply(prompt: str) -> str:
        text = format_prompt(prompt) if args.instruct else prompt
        out = generate_text(model, tokenizer, text, args.max_new_tokens,
                            args.temperature, args.top_k or None)
        return " " + extract_response(out) if args.instruct else out

    if args.prompt is not None:
        print(args.prompt + reply(args.prompt))
        return

    if args.instruct:
        print("Jarvis listo. Hazle una pregunta o dale una instrucción (Ctrl+C para salir).")
    else:
        print("Jarvis listo. Escribe el inicio de un texto y Jarvis lo continuará (Ctrl+C para salir).")
    try:
        while True:
            prompt = input("\nTú: ").strip()
            if prompt.lower() in {"salir", "exit", "quit"}:
                break
            if prompt:
                print(f"Jarvis:{reply(prompt)}" if args.instruct else f"Jarvis: {prompt}{reply(prompt)}")
    except (KeyboardInterrupt, EOFError):
        pass
    print("\nHasta luego.")


if __name__ == "__main__":
    main()
