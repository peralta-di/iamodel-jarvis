"""Motores ("cerebros") que Jarvis puede usar para generar las respuestas.

- ClaudeMotor: API de Anthropic. El más capaz: usa herramientas (cálculo exacto,
  búsqueda en documentos) y puede buscar y leer páginas web.
- OllamaMotor: modelos abiertos que se ejecutan gratis en tu ordenador con Ollama.
- JarvisMotor: el modelo que entrenaste desde cero (train.py + finetune.py).
  Es educativo: con unos pocos millones de parámetros no puede resolver tareas reales.

Todos comparten la misma interfaz, así el resto de la app no depende del motor.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from .herramientas import Herramientas

Progreso = Callable[[str], None]


class MotorNoDisponible(RuntimeError):
    """El motor no puede hacer esta tarea o no está configurado."""


@dataclass
class Respuesta:
    texto: str
    api: list[dict] = field(default_factory=list)      # mensajes nuevos, en formato de la API
    fuentes: list[dict] = field(default_factory=list)  # [{"titulo": ..., "url": ...}]
    herramientas: list[dict] = field(default_factory=list)


class Motor:
    nombre = "motor"
    usa_herramientas = False
    usa_web = False
    max_contexto_chars = 20_000  # cuánto texto cabe cómodamente en una petición

    def conversar(self, sistema: str, historial: list[dict], herramientas: Herramientas | None = None,
                  web: bool = False, al_progreso: Progreso | None = None) -> Respuesta:
        raise NotImplementedError

    def completar(self, prompt: str, sistema: str = "") -> str:
        return self.conversar(sistema, [{"role": "user", "content": prompt}]).texto

    def completar_json(self, prompt: str, esquema: dict, sistema: str = "") -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------------- Claude
class ClaudeMotor(Motor):
    nombre = "Claude"
    usa_herramientas = True
    usa_web = True
    max_contexto_chars = 600_000  # ~1M tokens de contexto; dejamos margen

    def __init__(self, api_key: str | None = None, modelo: str = "claude-opus-5-5",
                 esfuerzo: str = "medium", max_tokens: int = 16000, cliente=None):
        if cliente is None:
            import anthropic
            cliente = anthropic.Anthropic(api_key=api_key or None)
        self.cliente = cliente
        self.modelo = modelo
        self.esfuerzo = esfuerzo
        self.max_tokens = max_tokens

    def _crear(self, formato: dict | None = None, **kwargs):
        output_config = {"effort": self.esfuerzo}
        if formato:
            output_config["format"] = formato
        return self.cliente.beta.messages.create(
            model=self.modelo,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            output_config=output_config,
            # Si el modelo rechaza una petición, la API la reintenta con otro modelo.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            cache_control={"type": "ephemeral"},
            **kwargs,
        )

    def conversar(self, sistema, historial, herramientas=None, web=False, al_progreso=None):
        tools = list(herramientas.definiciones()) if herramientas else []
        if web:
            tools += [{"type": "web_search_20260209", "name": "web_search", "max_uses": 8},
                      {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 8}]
        kwargs = {"system": sistema} if sistema else {}
        if tools:
            kwargs["tools"] = tools

        nuevos: list[dict] = []
        fuentes: list[dict] = []
        respuesta = None
        for _ in range(15):  # límite de rondas de herramientas
            respuesta = self._crear(messages=list(historial) + nuevos, **kwargs)
            nuevos.append({"role": "assistant", "content": [b.to_dict() for b in respuesta.content]})
            fuentes += _fuentes_claude(respuesta.content)

            if respuesta.stop_reason == "pause_turn":  # búsqueda web larga: continuar
                if al_progreso:
                    al_progreso("Buscando en internet…")
                continue
            if respuesta.stop_reason != "tool_use":
                break
            resultados = []
            for bloque in respuesta.content:
                if bloque.type != "tool_use":
                    continue
                if al_progreso:
                    al_progreso(f"Usando la herramienta «{bloque.name}»…")
                if herramientas is None:
                    texto, error = "Herramienta no disponible.", True
                else:
                    texto, error = herramientas.ejecutar(bloque.name, dict(bloque.input or {}))
                resultados.append({"type": "tool_result", "tool_use_id": bloque.id,
                                   "content": texto, "is_error": error})
            nuevos.append({"role": "user", "content": resultados})

        texto = "".join(b.text for b in respuesta.content if b.type == "text").strip()
        if respuesta.stop_reason == "refusal":
            texto = texto or "Lo siento, no puedo ayudar con esa petición."
        elif respuesta.stop_reason == "max_tokens":
            texto += "\n\n*(Respuesta cortada por longitud. Pide que continúe.)*"
        return Respuesta(texto=texto, api=nuevos, fuentes=_sin_duplicados(fuentes),
                         herramientas=list(herramientas.registro) if herramientas else [])

    def completar_json(self, prompt, esquema, sistema=""):
        kwargs = {"system": sistema} if sistema else {}
        respuesta = self._crear(formato={"type": "json_schema", "schema": esquema},
                                messages=[{"role": "user", "content": prompt}], **kwargs)
        if respuesta.stop_reason == "refusal":
            raise MotorNoDisponible("El modelo rechazó la petición.")
        texto = next((b.text for b in respuesta.content if b.type == "text"), "")
        return json.loads(texto)


def _fuentes_claude(contenido) -> list[dict]:
    fuentes = []
    for bloque in contenido:
        if bloque.type == "web_search_tool_result" and isinstance(bloque.content, list):
            for r in bloque.content:
                if getattr(r, "url", None):
                    fuentes.append({"titulo": getattr(r, "title", "") or r.url, "url": r.url})
        elif bloque.type == "text":
            for c in getattr(bloque, "citations", None) or []:
                if getattr(c, "url", None):
                    fuentes.append({"titulo": getattr(c, "title", "") or c.url, "url": c.url})
    return fuentes


def _sin_duplicados(fuentes: list[dict]) -> list[dict]:
    vistas, out = set(), []
    for f in fuentes:
        if f["url"] not in vistas:
            vistas.add(f["url"])
            out.append(f)
    return out


# ---------------------------------------------------------------------- Ollama
class OllamaMotor(Motor):
    nombre = "Ollama"
    max_contexto_chars = 24_000

    def __init__(self, modelo: str = "llama3.1", url: str = "http://localhost:11434",
                 num_ctx: int = 16384):
        self.modelo = modelo
        self.url = url.rstrip("/")
        self.num_ctx = num_ctx

    def _chat(self, mensajes: list[dict], formato: dict | None = None) -> str:
        cuerpo = {"model": self.modelo, "messages": mensajes, "stream": False,
                  "options": {"num_ctx": self.num_ctx}}
        if formato:
            cuerpo["format"] = formato
        req = urllib.request.Request(f"{self.url}/api/chat", data=json.dumps(cuerpo).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read())["message"]["content"]
        except urllib.error.URLError as e:
            raise MotorNoDisponible(
                f"No se pudo conectar con Ollama en {self.url}. ¿Está instalado y en marcha? "
                f"(https://ollama.com, luego: ollama pull {self.modelo})") from e

    def conversar(self, sistema, historial, herramientas=None, web=False, al_progreso=None):
        mensajes = ([{"role": "system", "content": sistema}] if sistema else []) + _a_texto_plano(historial)
        texto = self._chat(mensajes).strip()
        return Respuesta(texto=texto, api=[{"role": "assistant", "content": texto}])

    def completar_json(self, prompt, esquema, sistema=""):
        mensajes = ([{"role": "system", "content": sistema}] if sistema else []) + [
            {"role": "user", "content": prompt + "\n\nResponde solo con JSON válido."}]
        return _extraer_json(self._chat(mensajes, formato=esquema))


# ---------------------------------------------------------------------- Jarvis
class JarvisMotor(Motor):
    nombre = "Jarvis (modelo propio)"
    max_contexto_chars = 300

    def __init__(self, ruta: str = "checkpoints/jarvis-chat", temperatura: float = 0.7):
        from jarvis import checkpoint
        if not os.path.exists(os.path.join(ruta, "model.pt")):
            raise MotorNoDisponible(
                f"No hay modelo en {ruta}. Entrénalo con train.py y finetune.py.")
        self.modelo, self.tokenizador = checkpoint.load(ruta)
        self.temperatura = temperatura

    def conversar(self, sistema, historial, herramientas=None, web=False, al_progreso=None):
        from jarvis.generate import generate_text
        from jarvis.instruct import extract_response, format_prompt
        ultimo = next((m["content"] for m in reversed(_a_texto_plano(historial)) if m["role"] == "user"), "")
        # El modelo propio tiene un contexto muy corto: solo ve la última petición.
        prompt = format_prompt(ultimo[-self.max_contexto_chars:])
        salida = generate_text(self.modelo, self.tokenizador, prompt, max_new_tokens=120,
                               temperature=self.temperatura, top_k=40)
        texto = extract_response(salida) or "(Jarvis no supo qué responder)"
        return Respuesta(texto=texto, api=[{"role": "assistant", "content": texto}])

    def completar_json(self, prompt, esquema, sistema=""):
        raise MotorNoDisponible("El modelo propio es demasiado pequeño para esta función. "
                                "Elige Claude u Ollama en la barra lateral.")


# ------------------------------------------------------------------- utilidades
def _a_texto_plano(historial: list[dict]) -> list[dict]:
    """Convierte mensajes con bloques (formato Claude) en mensajes de texto simples."""
    out: list[dict] = []
    for m in historial:
        if m["role"] not in ("user", "assistant"):
            continue
        contenido, imagenes = m["content"], []
        if isinstance(contenido, list):
            partes = []
            for b in contenido:
                if b.get("type") == "text":
                    partes.append(b["text"])
                elif b.get("type") == "image" and b.get("source", {}).get("type") == "base64":
                    imagenes.append(b["source"]["data"])
            contenido = "\n".join(partes)
        if not contenido.strip() and not imagenes:
            continue
        if out and out[-1]["role"] == m["role"]:
            out[-1]["content"] += "\n\n" + contenido
            out[-1].setdefault("images", []).extend(imagenes)
        else:
            out.append({"role": m["role"], "content": contenido, **({"images": imagenes} if imagenes else {})})
    return out


def _extraer_json(texto: str) -> dict:
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", texto, re.S)
        if not m:
            raise MotorNoDisponible("El modelo no devolvió JSON válido. Inténtalo de nuevo.")
        return json.loads(m.group(0))


def bloque_imagen(datos: bytes, tipo_mime: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": tipo_mime,
                                        "data": base64.standard_b64encode(datos).decode()}}
