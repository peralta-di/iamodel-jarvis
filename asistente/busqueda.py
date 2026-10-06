"""Buscador de fragmentos con BM25, escrito desde cero.

Es la parte de "recuperación" de RAG (Retrieval-Augmented Generation, la tercera
guía de la sección "Build your own AI Model"): antes de responder, Jarvis busca
los fragmentos de tus documentos más relacionados con la pregunta y se los da
al modelo como contexto, para que responda según tus textos.

BM25 puntúa cada fragmento según cuántas palabras de la consulta contiene,
dando más peso a las palabras raras y penalizando los fragmentos muy largos.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

STOPWORDS = set("""
a al algo algunas algunos ante antes como con contra cual cuando de del desde donde dos el
ella ellas ellos en entre era es esa esas ese eso esos esta estas este esto estos fue ha
han hay la las le les lo los mas me mi mis muy ni no nos o os otra otro para pero por porque
que se sea ser si sin sobre son su sus tambien te tiene tu un una uno unos y ya yo
the of and to in is it that for on as are with be this by an or from at
""".split())


def normalizar(texto: str) -> list[str]:
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")  # quita tildes
    palabras = re.findall(r"[a-z0-9ñ]+", texto)
    return [_raiz(p) for p in palabras if p not in STOPWORDS and len(p) > 1]


def _raiz(palabra: str) -> str:
    """Reducción muy simple de plurales y terminaciones frecuentes en español."""
    for suf in ("aciones", "ciones", "acion", "cion", "mente", "s"):
        if len(palabra) > len(suf) + 3 and palabra.endswith(suf):
            return palabra[: -len(suf)]
    return palabra


@dataclass
class Resultado:
    id: int
    puntuacion: float
    texto: str
    documento: str
    pagina: int | None


class BM25:
    def __init__(self, fragmentos: list[dict], k1: float = 1.5, b: float = 0.75):
        """`fragmentos`: dicts con 'id', 'texto', 'documento' y 'pagina'."""
        self.fragmentos = fragmentos
        self.k1, self.b = k1, b
        self.tokens = [Counter(normalizar(f["texto"])) for f in fragmentos]
        self.longitudes = [sum(t.values()) for t in self.tokens]
        self.media = sum(self.longitudes) / max(len(self.longitudes), 1)
        df = Counter(term for t in self.tokens for term in t)
        n = len(fragmentos)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def buscar(self, consulta: str, k: int = 5) -> list[Resultado]:
        terminos = normalizar(consulta)
        puntuados = []
        for i, tf in enumerate(self.tokens):
            score = 0.0
            for term in terminos:
                if term in tf:
                    f = tf[term]
                    norm = 1 - self.b + self.b * self.longitudes[i] / (self.media or 1)
                    score += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * norm)
            if score > 0:
                puntuados.append((score, i))
        puntuados.sort(reverse=True)
        return [Resultado(self.fragmentos[i]["id"], s, self.fragmentos[i]["texto"],
                          self.fragmentos[i]["documento"], self.fragmentos[i]["pagina"])
                for s, i in puntuados[:k]]


def formatear_contexto(resultados: list[Resultado]) -> str:
    """Convierte los resultados en un bloque de fuentes numeradas [1], [2]..."""
    bloques = []
    for n, r in enumerate(resultados, 1):
        donde = f"{r.documento}, pág. {r.pagina}" if r.pagina else r.documento
        bloques.append(f"[{n}] ({donde})\n{r.texto}")
    return "\n\n".join(bloques)
