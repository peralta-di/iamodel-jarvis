"""Descarga libros de dominio público del Proyecto Gutenberg a data/.

    python scripts/download_data.py                  # Don Quijote (libro 2000)
    python scripts/download_data.py --id 1234 --name mi_libro

Busca más libros en español en https://www.gutenberg.org/ebooks/search/?query=l.es
(el número que aparece en la URL del libro es su --id).
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def strip_gutenberg(text: str) -> str:
    """Quita la cabecera y el pie de licencia de Gutenberg."""
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text)
    return text[start.end() if start else 0:end.start() if end else len(text)].strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--id", type=int, default=2000, help="número del libro en Gutenberg")
    p.add_argument("--name", default=None, help="nombre del archivo (sin .txt)")
    args = p.parse_args()

    name = args.name or ("quijote" if args.id == 2000 else f"gutenberg_{args.id}")
    url = f"https://www.gutenberg.org/cache/epub/{args.id}/pg{args.id}.txt"
    dest = DATA_DIR / f"{name}.txt"
    print(f"Descargando {url} ...")
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8-sig")
    DATA_DIR.mkdir(exist_ok=True)
    dest.write_text(strip_gutenberg(text), encoding="utf-8")
    print(f"Guardado en {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
