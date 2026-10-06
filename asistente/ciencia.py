"""Herramientas exactas de matemáticas y química.

Los modelos de lenguaje se equivocan con frecuencia en cálculos largos, así que
Jarvis delega las cuentas en estas funciones (álgebra simbólica con SymPy,
masas molares y ajuste de ecuaciones químicas) y solo explica el resultado.
"""

from __future__ import annotations

import re
from collections import Counter

import sympy as sp
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                        parse_expr, standard_transformations)

TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_FORBIDDEN = re.compile(r"__|\bimport\b|\blambda\b|\bexec\b|\beval\b|\bopen\b|[;`]")

OPERACIONES = ["evaluar", "simplificar", "expandir", "factorizar", "resolver", "derivar",
               "integrar", "limite", "serie", "matriz"]


def _parse(text: str, variables: dict[str, sp.Symbol] | None = None) -> sp.Expr:
    text = text.strip()
    if not text or len(text) > 500 or _FORBIDDEN.search(text):
        raise ValueError("Expresión vacía, demasiado larga o con caracteres no permitidos.")
    local = {"e": sp.E, "pi": sp.pi, "oo": sp.oo, "i": sp.I, "ln": sp.log, "sen": sp.sin,
             "raiz": sp.sqrt, **(variables or {})}
    return parse_expr(text.replace("√", "sqrt"), local_dict=local, transformations=TRANSFORMS)


def _symbols(expr) -> list[sp.Symbol]:
    return sorted(getattr(expr, "free_symbols", set()), key=lambda s: s.name)


def _fmt(value) -> dict:
    out = {"resultado": str(value), "latex": sp.latex(value)}
    try:
        if isinstance(value, sp.Basic) and value.is_number and not value.is_Integer:
            out["aproximado"] = str(sp.N(value, 10))
    except (TypeError, ValueError):
        pass
    return out


def calcular(operacion: str, expresion: str, variable: str = "", desde: str = "",
             hasta: str = "", orden: int = 1) -> dict:
    """Ejecuta una operación matemática exacta.

    - resolver: `expresion` puede tener '=' y varias ecuaciones separadas por ','.
    - integrar: con `desde`/`hasta` calcula la integral definida.
    - limite: `hasta` es el punto al que tiende la variable.
    - serie: desarrollo de Taylor alrededor de `desde` (0 por defecto) hasta `orden`.
    - matriz: `expresion` como [[1,2],[3,4]]; devuelve determinante, inversa y autovalores.
    """
    op = operacion.lower().strip()
    if op not in OPERACIONES:
        raise ValueError(f"Operación desconocida '{operacion}'. Usa una de: {', '.join(OPERACIONES)}")
    var = sp.Symbol(variable) if variable else None

    if op == "resolver":
        eqs = []
        for part in re.split(r",(?![^\[\(]*[\]\)])", expresion):
            if "=" in part:
                left, right = part.split("=", 1)
                eqs.append(sp.Eq(_parse(left), _parse(right)))
            else:
                eqs.append(sp.Eq(_parse(part), 0))
        unknowns = [sp.Symbol(v.strip()) for v in variable.split(",")] if variable else \
            sorted(set().union(*(e.free_symbols for e in eqs)), key=lambda s: s.name)
        sol = sp.solve(eqs if len(eqs) > 1 else eqs[0], unknowns, dict=True)
        res = {"resultado": str(sol) if sol else "Sin solución",
               "latex": ", ".join(sp.latex(s) for s in sol)}
        approx = [{str(k): str(sp.N(v, 8)) for k, v in s.items()} for s in sol
                  if all(getattr(v, "is_number", False) for v in s.values())]
        if approx:
            res["aproximado"] = str(approx)
        return res

    if op == "matriz":
        m = sp.Matrix(_parse(expresion))
        res = {"matriz": str(m), "latex": sp.latex(m)}
        if m.is_square:
            det = m.det()
            res["determinante"] = str(det)
            res["autovalores"] = str(m.eigenvals())
            if det != 0:
                res["inversa"] = str(m.inv())
        res["rango"] = str(m.rank())
        return res

    expr = _parse(expresion)
    var = var or (_symbols(expr)[0] if _symbols(expr) else sp.Symbol("x"))
    if op == "evaluar":
        value = sp.simplify(expr)
    elif op == "simplificar":
        value = sp.simplify(expr)
    elif op == "expandir":
        value = sp.expand(expr)
    elif op == "factorizar":
        value = sp.factor(expr)
    elif op == "derivar":
        value = sp.diff(expr, var, int(orden))
    elif op == "integrar":
        value = sp.integrate(expr, (var, _parse(desde), _parse(hasta))) if desde and hasta \
            else sp.integrate(expr, var)
    elif op == "limite":
        value = sp.limit(expr, var, _parse(hasta or "0"))
    else:  # serie
        value = sp.series(expr, var, _parse(desde or "0"), int(orden) + 1).removeO()
    return _fmt(value)


# ------------------------------------------------------------------- química
MASAS = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305, "Al": 26.982,
    "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098,
    "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723,
    "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904, "Kr": 83.798, "Rb": 85.468,
    "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.95, "Tc": 98.0,
    "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82,
    "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29, "Cs": 132.91,
    "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24, "Sm": 150.36,
    "Eu": 151.96, "Gd": 157.25, "Tb": 158.93, "Dy": 162.50, "Ho": 164.93, "Er": 167.26,
    "Tm": 168.93, "Yb": 173.05, "Lu": 174.97, "Hf": 178.49, "Ta": 180.95, "W": 183.84,
    "Re": 186.21, "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Po": 209.0, "At": 210.0, "Rn": 222.0,
    "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.04, "Pa": 231.04, "U": 238.03,
    "Np": 237.0, "Pu": 244.0,
}


def contar_atomos(formula: str) -> Counter:
    """'Ca(OH)2' -> {'Ca': 1, 'O': 2, 'H': 2}. Admite paréntesis, corchetes y hidratos con '·'."""
    formula = formula.replace(" ", "").replace("[", "(").replace("]", ")")
    total: Counter = Counter()
    for part in re.split(r"[·*.]", formula):
        m = re.match(r"^(\d+)(.*)$", part)
        mult, part = (int(m.group(1)), m.group(2)) if m else (1, part)
        for el, n in _parse_formula(part).items():
            total[el] += n * mult
    return total


def _parse_formula(formula: str) -> Counter:
    stack: list[Counter] = [Counter()]
    for token in re.finditer(r"([A-Z][a-z]?)(\d*)|(\()|(\))(\d*)|(\S)", formula):
        el, n, open_, close, close_n, bad = token.groups()
        if bad:
            raise ValueError(f"Carácter no válido en la fórmula: '{bad}'")
        if el:
            if el not in MASAS:
                raise ValueError(f"Elemento desconocido: {el}")
            stack[-1][el] += int(n or 1)
        elif open_:
            stack.append(Counter())
        elif close:
            if len(stack) == 1:
                raise ValueError("Paréntesis sin abrir")
            group = stack.pop()
            for k, v in group.items():
                stack[-1][k] += v * int(close_n or 1)
    if len(stack) != 1:
        raise ValueError("Paréntesis sin cerrar")
    return stack[0]


def masa_molar(formula: str) -> dict:
    atoms = contar_atomos(formula)
    detalle = {el: f"{n} × {MASAS[el]} = {n * MASAS[el]:.3f}" for el, n in atoms.items()}
    total = sum(n * MASAS[el] for el, n in atoms.items())
    return {"formula": formula, "masa_molar_g_mol": round(total, 3), "detalle": detalle}


def balancear_ecuacion(ecuacion: str) -> dict:
    """'H2 + O2 -> H2O' -> '2 H2 + O2 -> 2 H2O'."""
    sides = re.split(r"->|→|=", ecuacion)
    if len(sides) != 2:
        raise ValueError("Escribe la ecuación como 'reactivos -> productos'.")
    reactivos = [s.strip() for s in sides[0].split("+") if s.strip()]
    productos = [s.strip() for s in sides[1].split("+") if s.strip()]
    especies = reactivos + productos
    counts = [contar_atomos(s) for s in especies]
    elementos = sorted(set().union(*counts))
    matrix = sp.Matrix([[c[el] * (1 if i < len(reactivos) else -1) for i, c in enumerate(counts)]
                        for el in elementos])
    null = matrix.nullspace()
    if len(null) != 1:
        raise ValueError("La ecuación no se puede ajustar de forma única (revisa las fórmulas).")
    vec = null[0]
    vec = vec * sp.ilcm(*[term.q for term in vec])
    if any(v < 0 for v in vec):
        vec = -vec
    if any(v <= 0 for v in vec):
        raise ValueError("No existe un ajuste con coeficientes positivos.")
    coef = [int(v) for v in vec]
    fmt = lambda c, s: s if c == 1 else f"{c} {s}"
    lados = (" + ".join(fmt(c, s) for c, s in zip(coef[:len(reactivos)], reactivos)),
             " + ".join(fmt(c, s) for c, s in zip(coef[len(reactivos):], productos)))
    return {"ecuacion_ajustada": f"{lados[0]} -> {lados[1]}",
            "coeficientes": dict(zip(especies, coef))}
