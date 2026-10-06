"""Guarda chats, mensajes y documentos en una base de datos SQLite local."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .busqueda import BM25
from .documentos import Pagina, fragmentar

ESQUEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    titulo TEXT NOT NULL,
    creado REAL NOT NULL,
    actualizado REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS mensajes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    rol TEXT NOT NULL,              -- 'user' o 'assistant'
    texto TEXT NOT NULL,            -- lo que se muestra en pantalla
    api TEXT,                       -- mensajes exactos enviados/recibidos del modelo (JSON)
    extra TEXT,                     -- fuentes, modelo usado, etc. (JSON)
    creado REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS documentos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    nombre TEXT NOT NULL,
    origen TEXT NOT NULL,           -- 'archivo' o la URL
    paginas INTEGER,
    caracteres INTEGER NOT NULL,
    creado REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS fragmentos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    documento_id INTEGER NOT NULL REFERENCES documentos(id) ON DELETE CASCADE,
    pagina INTEGER,
    texto TEXT NOT NULL
);
"""


class Almacen:
    def __init__(self, ruta: str | Path = "jarvis_datos/jarvis.db"):
        ruta = Path(ruta)
        if str(ruta) != ":memory:":
            ruta.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(ruta), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(ESQUEMA)
        self._indices: dict[int, tuple[int, BM25]] = {}

    # ------------------------------------------------------------------ chats
    def crear_chat(self, titulo: str = "Nuevo chat") -> int:
        now = time.time()
        cur = self.db.execute("INSERT INTO chats (titulo, creado, actualizado) VALUES (?, ?, ?)",
                              (titulo, now, now))
        self.db.commit()
        return cur.lastrowid

    def listar_chats(self) -> list[dict]:
        rows = self.db.execute("SELECT * FROM chats ORDER BY actualizado DESC").fetchall()
        return [dict(r) for r in rows]

    def renombrar_chat(self, chat_id: int, titulo: str) -> None:
        self.db.execute("UPDATE chats SET titulo = ? WHERE id = ?", (titulo, chat_id))
        self.db.commit()

    def borrar_chat(self, chat_id: int) -> None:
        self.db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        self.db.commit()
        self._indices.pop(chat_id, None)

    # --------------------------------------------------------------- mensajes
    def agregar_mensaje(self, chat_id: int, rol: str, texto: str, api: list | None = None,
                        extra: dict | None = None) -> int:
        now = time.time()
        cur = self.db.execute(
            "INSERT INTO mensajes (chat_id, rol, texto, api, extra, creado) VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, rol, texto, json.dumps(api, ensure_ascii=False) if api is not None else None,
             json.dumps(extra or {}, ensure_ascii=False), now))
        self.db.execute("UPDATE chats SET actualizado = ? WHERE id = ?", (now, chat_id))
        self.db.commit()
        return cur.lastrowid

    def mensajes(self, chat_id: int) -> list[dict]:
        rows = self.db.execute("SELECT * FROM mensajes WHERE chat_id = ? ORDER BY id",
                               (chat_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["api"] = json.loads(d["api"]) if d["api"] else None
            d["extra"] = json.loads(d["extra"]) if d["extra"] else {}
            out.append(d)
        return out

    # ------------------------------------------------------------- documentos
    def agregar_documento(self, chat_id: int, nombre: str, paginas: list[Pagina],
                          origen: str = "archivo") -> int:
        trozos = fragmentar(paginas)
        if not trozos:
            raise ValueError(f"'{nombre}' no contiene texto.")
        cur = self.db.execute(
            "INSERT INTO documentos (chat_id, nombre, origen, paginas, caracteres, creado) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, nombre, origen, sum(1 for p in paginas if p.numero) or None,
             sum(len(p.texto) for p in paginas), time.time()))
        doc_id = cur.lastrowid
        self.db.executemany("INSERT INTO fragmentos (documento_id, pagina, texto) VALUES (?, ?, ?)",
                            [(doc_id, pag, txt) for pag, txt in trozos])
        self.db.commit()
        self._indices.pop(chat_id, None)
        return doc_id

    def documentos(self, chat_id: int) -> list[dict]:
        rows = self.db.execute("SELECT * FROM documentos WHERE chat_id = ? ORDER BY id",
                               (chat_id,)).fetchall()
        return [dict(r) for r in rows]

    def borrar_documento(self, doc_id: int) -> None:
        row = self.db.execute("SELECT chat_id FROM documentos WHERE id = ?", (doc_id,)).fetchone()
        self.db.execute("DELETE FROM documentos WHERE id = ?", (doc_id,))
        self.db.commit()
        if row:
            self._indices.pop(row["chat_id"], None)

    def texto_documento(self, doc_id: int) -> str:
        """Texto completo del documento (los fragmentos se unen sin el solape)."""
        rows = self.db.execute("SELECT pagina, texto FROM fragmentos WHERE documento_id = ? ORDER BY id",
                               (doc_id,)).fetchall()
        partes, ultima_pagina = [], object()
        for r in rows:
            if r["pagina"] != ultima_pagina and r["pagina"] is not None:
                partes.append(f"\n[Página {r['pagina']}]\n")
                ultima_pagina = r["pagina"]
            partes.append(_quitar_solape(partes[-1] if partes else "", r["texto"]))
        return "".join(partes).strip()

    def indice(self, chat_id: int) -> BM25:
        """Índice de búsqueda de todos los documentos del chat (se cachea)."""
        total = self.db.execute(
            "SELECT COUNT(*) FROM fragmentos f JOIN documentos d ON d.id = f.documento_id "
            "WHERE d.chat_id = ?", (chat_id,)).fetchone()[0]
        cached = self._indices.get(chat_id)
        if cached and cached[0] == total:
            return cached[1]
        rows = self.db.execute(
            "SELECT f.id, f.texto, f.pagina, d.nombre AS documento FROM fragmentos f "
            "JOIN documentos d ON d.id = f.documento_id WHERE d.chat_id = ? ORDER BY f.id",
            (chat_id,)).fetchall()
        idx = BM25([dict(r) for r in rows])
        self._indices[chat_id] = (total, idx)
        return idx


def _quitar_solape(anterior: str, actual: str, maximo: int = 400) -> str:
    """Si `actual` empieza repitiendo el final de `anterior`, quita esa repetición."""
    for n in range(min(maximo, len(anterior), len(actual)), 20, -1):
        if anterior.endswith(actual[:n]):
            return actual[n:]
    return ("\n" if anterior and not anterior.endswith("\n") else "") + actual
