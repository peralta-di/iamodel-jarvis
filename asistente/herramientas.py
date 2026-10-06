"""Herramientas que el modelo puede usar mientras responde (tool use)."""

from __future__ import annotations

import json

from . import ciencia
from .busqueda import BM25, formatear_contexto

DEFINICIONES = [
    {
        "name": "calcular",
        "description": (
            "Cálculo matemático exacto con SymPy. Úsala SIEMPRE para operaciones algebraicas, "
            "ecuaciones, derivadas, integrales, límites, series, matrices o cuentas numéricas "
            "en lugar de calcular de cabeza. Sintaxis de Python/SymPy: x**2 o x^2, sqrt(x), "
            "sin(x), exp(x), log(x), pi, E."),
        "input_schema": {
            "type": "object",
            "properties": {
                "operacion": {"type": "string", "enum": ciencia.OPERACIONES},
                "expresion": {"type": "string", "description":
                              "Expresión. Para 'resolver' puede incluir '=' y varias ecuaciones "
                              "separadas por comas. Para 'matriz': [[1,2],[3,4]]."},
                "variable": {"type": "string", "description": "Variable(s), p. ej. 'x' o 'x,y'."},
                "desde": {"type": "string", "description": "Límite inferior de integración o punto de la serie."},
                "hasta": {"type": "string", "description": "Límite superior de integración o punto del límite."},
                "orden": {"type": "integer", "description": "Orden de la derivada o de la serie."},
            },
            "required": ["operacion", "expresion"],
        },
    },
    {
        "name": "masa_molar",
        "description": "Calcula la masa molar (g/mol) de una fórmula química, p. ej. 'H2SO4' o 'CuSO4·5H2O'.",
        "input_schema": {
            "type": "object",
            "properties": {"formula": {"type": "string"}},
            "required": ["formula"],
        },
    },
    {
        "name": "balancear_ecuacion",
        "description": "Ajusta una ecuación química, p. ej. 'C3H8 + O2 -> CO2 + H2O'.",
        "input_schema": {
            "type": "object",
            "properties": {"ecuacion": {"type": "string"}},
            "required": ["ecuacion"],
        },
    },
]

BUSCAR_DOCUMENTOS = {
    "name": "buscar_documentos",
    "description": ("Busca en los documentos que el usuario ha subido (PDFs, apuntes, páginas web "
                    "importadas) y devuelve los fragmentos más relevantes con su documento y página."),
    "input_schema": {
        "type": "object",
        "properties": {
            "consulta": {"type": "string", "description": "Palabras clave o pregunta a buscar."},
            "cantidad": {"type": "integer", "description": "Número de fragmentos (1-10)."},
        },
        "required": ["consulta"],
    },
}


class Herramientas:
    def __init__(self, indice: BM25 | None = None):
        self.indice = indice if indice and indice.fragmentos else None
        self.registro: list[dict] = []  # historial de llamadas, para mostrarlo en pantalla

    def definiciones(self) -> list[dict]:
        return DEFINICIONES + ([BUSCAR_DOCUMENTOS] if self.indice else [])

    def ejecutar(self, nombre: str, entrada: dict) -> tuple[str, bool]:
        """Devuelve (resultado, es_error)."""
        try:
            if nombre == "calcular":
                res = ciencia.calcular(
                    str(entrada["operacion"]), str(entrada["expresion"]), str(entrada.get("variable") or ""),
                    str(entrada.get("desde") or ""), str(entrada.get("hasta") or ""),
                    int(entrada.get("orden") or 1))
            elif nombre == "masa_molar":
                res = ciencia.masa_molar(str(entrada["formula"]))
            elif nombre == "balancear_ecuacion":
                res = ciencia.balancear_ecuacion(str(entrada["ecuacion"]))
            elif nombre == "buscar_documentos" and self.indice:
                k = max(1, min(10, int(entrada.get("cantidad") or 5)))
                resultados = self.indice.buscar(str(entrada["consulta"]), k)
                res = formatear_contexto(resultados) if resultados else "No se encontraron fragmentos."
            else:
                raise ValueError(f"Herramienta desconocida: {nombre}")
            texto = res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
            self.registro.append({"nombre": nombre, "entrada": entrada, "resultado": texto})
            return texto, False
        except Exception as e:  # el error se devuelve al modelo para que lo corrija
            msg = f"Error: {e}"
            self.registro.append({"nombre": nombre, "entrada": entrada, "resultado": msg})
            return msg, True
