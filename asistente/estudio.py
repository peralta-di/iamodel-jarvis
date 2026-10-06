"""Funciones de estudio de Jarvis: chat, resúmenes, explicaciones, quizzes,
tarjetas, mapas mentales, diagramas, presentaciones e investigación profunda."""

from __future__ import annotations

from .almacen import Almacen
from .busqueda import BM25, formatear_contexto
from .herramientas import Herramientas
from .motores import Motor, Progreso, Respuesta

SISTEMA = """Eres Jarvis, un asistente de estudio paciente y riguroso que responde en español.

Cómo trabajas:
- Explica paso a paso y con claridad; adapta el nivel a la persona.
- En problemas de matemáticas, física o química: identifica los datos, plantea las fórmulas, \
resuelve paso a paso indicando unidades y comprueba el resultado. Usa la herramienta `calcular` \
(y `masa_molar` o `balancear_ecuacion` en química) para cualquier cuenta en lugar de calcular de cabeza.
- Escribe las fórmulas en LaTeX entre $...$ (en línea) o $$...$$ (en bloque).
- Cuando uses fragmentos de los documentos del usuario, cita la fuente con [n].
- Si no estás seguro de algo, dilo. No inventes datos ni referencias."""

SOLO_DOCUMENTOS = """

MODO «SOLO MIS DOCUMENTOS»: responde únicamente con la información de los fragmentos de los \
documentos del usuario (los que aparecen en el mensaje o los que encuentres con `buscar_documentos`). \
Cita cada afirmación con [n]. Si la respuesta no está en los documentos, dilo claramente y no \
uses conocimientos externos."""


def _con_contexto(pregunta: str, indice: BM25 | None, k: int = 6,
                  consulta: str | None = None) -> tuple[str, list]:
    """Añade a la pregunta los fragmentos más relevantes de los documentos (RAG)."""
    if not indice or not indice.fragmentos:
        return pregunta, []
    resultados = indice.buscar(consulta or pregunta, k)
    if not resultados:
        return pregunta, []
    return (f"Fragmentos relevantes de mis documentos:\n\n{formatear_contexto(resultados)}\n\n"
            f"---\n\n{pregunta}"), resultados


def historial_api(almacen: Almacen, chat_id: int) -> list[dict]:
    """Reconstruye los mensajes exactos de la conversación para enviarlos al modelo."""
    historial = []
    for m in almacen.mensajes(chat_id):
        historial.extend(m["api"] if m["api"] else [{"role": m["rol"], "content": m["texto"]}])
    return historial


# ------------------------------------------------------------------------ chat
def responder(motor: Motor, almacen: Almacen, chat_id: int, pregunta: str,
              solo_documentos: bool = False, web: bool = False, imagenes: list[dict] | None = None,
              al_progreso: Progreso | None = None) -> Respuesta:
    indice = almacen.indice(chat_id)
    texto_usuario, encontrados = _con_contexto(pregunta, indice)
    contenido = [*(imagenes or []), {"type": "text", "text": texto_usuario}]
    mensaje = {"role": "user", "content": contenido}

    sistema = SISTEMA + (SOLO_DOCUMENTOS if solo_documentos else "")
    herramientas = Herramientas(indice)
    respuesta = motor.conversar(sistema, historial_api(almacen, chat_id) + [mensaje], herramientas,
                                web=web and not solo_documentos, al_progreso=al_progreso)

    documentos = [{"documento": r.documento, "pagina": r.pagina, "texto": r.texto} for r in encontrados]
    almacen.agregar_mensaje(chat_id, "user", pregunta, api=[mensaje],
                            extra={"imagenes": len(imagenes or [])})
    almacen.agregar_mensaje(chat_id, "assistant", respuesta.texto, api=respuesta.api,
                            extra={"motor": motor.nombre, "fuentes": respuesta.fuentes,
                                   "documentos": documentos, "herramientas": respuesta.herramientas})
    if len(almacen.mensajes(chat_id)) == 2:  # primer intercambio: título automático
        almacen.renombrar_chat(chat_id, pregunta.strip().splitlines()[0][:50] or "Chat")
    return respuesta


# ------------------------------------------------------------------- resúmenes
FORMATOS_RESUMEN = {
    "Resumen breve": "Un resumen breve de un párrafo (5-8 líneas) con la idea central.",
    "Resumen detallado": "Un resumen detallado organizado con títulos (##) por secciones, "
                         "conservando definiciones, datos y fórmulas importantes.",
    "Puntos clave": "Una lista de 8-15 puntos clave, cada uno en una línea, empezando por lo más importante.",
    "Esquema jerárquico": "Un esquema jerárquico con numeración (1., 1.1., 1.1.1.) de los temas y subtemas.",
    "Método Cornell": "Notas en formato Cornell: una tabla Markdown con columnas «Preguntas clave» y "
                      "«Notas», y al final una sección «Resumen» de 3-4 líneas.",
    "Tabla comparativa": "Una o varias tablas Markdown que comparen los conceptos, elementos o "
                         "posturas del texto según criterios relevantes.",
    "Glosario": "Un glosario en orden alfabético con los términos importantes y su definición breve.",
    "Fórmulas y procedimientos": "Una hoja de fórmulas: cada fórmula en LaTeX con el significado de "
                                 "cada símbolo, sus unidades y cuándo se usa; y los procedimientos paso a paso.",
    "Línea de tiempo": "Una línea de tiempo en orden cronológico (fecha — acontecimiento — importancia).",
    "Explicación para niños": "Un resumen muy sencillo, como para un niño de 10 años, con ejemplos cotidianos.",
}


def resumir(motor: Motor, texto: str, formato: str, titulo: str = "el texto",
            instrucciones: str = "", al_progreso: Progreso | None = None) -> str:
    """Resume en el formato elegido. Si el texto no cabe en el modelo, primero lo
    resume por partes y después combina las partes (map-reduce)."""
    descripcion = FORMATOS_RESUMEN.get(formato, formato)
    limite = int(motor.max_contexto_chars * 0.8)
    if len(texto) > limite:
        partes = [texto[i:i + limite] for i in range(0, len(texto), limite)]
        notas = []
        for n, parte in enumerate(partes, 1):
            if al_progreso:
                al_progreso(f"Leyendo la parte {n} de {len(partes)}…")
            notas.append(motor.completar(
                f"Toma notas detalladas (ideas, definiciones, datos, fórmulas) de esta parte "
                f"{n}/{len(partes)} de {titulo}:\n\n{parte}", SISTEMA))
        texto = "\n\n".join(f"### Parte {n}\n{t}" for n, t in enumerate(notas, 1))
        if al_progreso:
            al_progreso("Combinando las partes…")
    extra = f"\nIndicaciones adicionales: {instrucciones}" if instrucciones.strip() else ""
    return motor.completar(
        f"Crea a partir de {titulo} lo siguiente: {descripcion}{extra}\n"
        f"Usa Markdown y no añadas información que no esté en el texto.\n\n<texto>\n{texto}\n</texto>",
        SISTEMA)


# ---------------------------------------------------------------- explicaciones
NIVELES = {
    "Niño (10 años)": "un niño de 10 años: lenguaje muy sencillo, analogías cotidianas, sin fórmulas",
    "Secundaria": "un estudiante de secundaria: lenguaje claro y fórmulas básicas explicadas",
    "Universidad": "un estudiante universitario: rigor, fórmulas y demostraciones breves",
    "Experto": "un experto: precisión técnica, matices y referencias a temas avanzados",
}


def explicar(motor: Motor, concepto: str, nivel: str, indice: BM25 | None = None,
             al_progreso: Progreso | None = None) -> Respuesta:
    publico = NIVELES.get(nivel, nivel)
    pedido = (f"Explícame «{concepto}» para {publico}. Estructura la explicación así:\n"
              "## Idea en una frase\n## Explicación\n## Analogía o intuición\n"
              "## Ejemplo resuelto (paso a paso; si hay cálculos, usa la herramienta `calcular`)\n"
              "## Errores comunes\n## Comprueba que lo entendiste (3 preguntas con su respuesta oculta "
              "dentro de <details><summary>Ver respuesta</summary>...</details>)")
    texto, _ = _con_contexto(pedido, indice, consulta=concepto)
    return motor.conversar(SISTEMA, [{"role": "user", "content": texto}], Herramientas(indice),
                           al_progreso=al_progreso)


# --------------------------------------------------------- quizzes y tarjetas
ESQUEMA_QUIZ = {
    "type": "object",
    "properties": {
        "titulo": {"type": "string"},
        "preguntas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tipo": {"type": "string", "enum": ["opcion_multiple", "verdadero_falso", "respuesta_corta"]},
                    "pregunta": {"type": "string"},
                    "opciones": {"type": "array", "items": {"type": "string"}},
                    "respuesta_correcta": {"type": "string"},
                    "explicacion": {"type": "string"},
                },
                "required": ["tipo", "pregunta", "opciones", "respuesta_correcta", "explicacion"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["titulo", "preguntas"],
    "additionalProperties": False,
}

TIPOS_PREGUNTA = {"Opción múltiple": "opcion_multiple", "Verdadero o falso": "verdadero_falso",
                  "Respuesta corta": "respuesta_corta"}


def _material(tema: str, texto: str, motor: Motor) -> str:
    if texto:
        return f"sobre el siguiente material:\n\n<material>\n{texto[:int(motor.max_contexto_chars * 0.8)]}\n</material>"
    return f"sobre el tema: {tema}"


def generar_quiz(motor: Motor, tema: str = "", texto: str = "", cantidad: int = 8,
                 dificultad: str = "media", tipos: list[str] | None = None) -> dict:
    tipos = [TIPOS_PREGUNTA.get(t, t) for t in (tipos or ["opcion_multiple"])]
    quiz = motor.completar_json(
        f"Crea un quiz de {cantidad} preguntas de dificultad {dificultad} {_material(tema, texto, motor)}\n\n"
        f"Tipos de pregunta permitidos: {', '.join(tipos)}.\n"
        "- opcion_multiple: 4 opciones plausibles; respuesta_correcta es exactamente el texto de una opción.\n"
        "- verdadero_falso: opciones ['Verdadero', 'Falso'].\n"
        "- respuesta_corta: opciones vacía; respuesta_correcta es la respuesta modelo.\n"
        "Varía la posición de la opción correcta. En explicacion, explica por qué es correcta "
        "(y por qué fallan las demás si aplica). Si el material es de un documento, pregunta solo por su contenido.",
        ESQUEMA_QUIZ, SISTEMA)
    quiz["preguntas"] = [p for p in quiz.get("preguntas", []) if p.get("pregunta")]
    return quiz


ESQUEMA_CORRECCION = {
    "type": "object",
    "properties": {"correcta": {"type": "boolean"}, "comentario": {"type": "string"}},
    "required": ["correcta", "comentario"],
    "additionalProperties": False,
}


def corregir_respuesta(motor: Motor, pregunta: str, modelo: str, respuesta: str) -> dict:
    return motor.completar_json(
        f"Pregunta: {pregunta}\nRespuesta modelo: {modelo}\nRespuesta del estudiante: {respuesta}\n\n"
        "¿Es correcta la respuesta del estudiante en lo esencial (aunque use otras palabras)? "
        "Da un comentario breve y constructivo.", ESQUEMA_CORRECCION, SISTEMA)


ESQUEMA_TARJETAS = {
    "type": "object",
    "properties": {"tarjetas": {"type": "array", "items": {
        "type": "object",
        "properties": {"frente": {"type": "string"}, "reverso": {"type": "string"}},
        "required": ["frente", "reverso"], "additionalProperties": False}}},
    "required": ["tarjetas"],
    "additionalProperties": False,
}


def generar_tarjetas(motor: Motor, tema: str = "", texto: str = "", cantidad: int = 15) -> list[dict]:
    datos = motor.completar_json(
        f"Crea {cantidad} tarjetas de estudio (flashcards) {_material(tema, texto, motor)}\n"
        "El frente es una pregunta o término concreto; el reverso, una respuesta breve y precisa.",
        ESQUEMA_TARJETAS, SISTEMA)
    return datos.get("tarjetas", [])


# ------------------------------------------------- mapas, diagramas y diapositivas
ESQUEMA_MAPA = {
    "type": "object",
    "properties": {
        "titulo": {"type": "string"},
        "ramas": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "titulo": {"type": "string"},
                "subramas": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"titulo": {"type": "string"},
                                   "detalles": {"type": "array", "items": {"type": "string"}}},
                    "required": ["titulo", "detalles"], "additionalProperties": False}},
            },
            "required": ["titulo", "subramas"], "additionalProperties": False}},
    },
    "required": ["titulo", "ramas"],
    "additionalProperties": False,
}


def generar_mapa_mental(motor: Motor, tema: str = "", texto: str = "") -> dict:
    return motor.completar_json(
        f"Crea un mapa mental {_material(tema, texto, motor)}\n"
        "Concepto central en titulo; 4-7 ramas principales; 2-4 subramas por rama; 0-3 detalles "
        "muy cortos (máx. 6 palabras) por subrama. Textos breves, sin frases largas.",
        ESQUEMA_MAPA, SISTEMA)


TIPOS_DIAGRAMA = {
    "Diagrama de flujo": "un diagrama de flujo de un proceso o algoritmo (inicio y fin en óvalo, "
                         "decisiones en rombo con conexiones etiquetadas Sí/No)",
    "Mapa conceptual": "un mapa conceptual con conceptos unidos por conexiones con verbos de enlace "
                       "como etiqueta (p. ej. «produce», «está formado por»)",
    "Ciclo": "un diagrama de ciclo cuyos pasos vuelven al inicio",
    "Jerarquía": "un diagrama jerárquico (árbol de clasificación) de arriba abajo",
}

ESQUEMA_DIAGRAMA = {
    "type": "object",
    "properties": {
        "titulo": {"type": "string"},
        "nodos": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "texto": {"type": "string"},
                           "forma": {"type": "string", "enum": ["caja", "rombo", "ovalo", "circulo"]}},
            "required": ["id", "texto", "forma"], "additionalProperties": False}},
        "conexiones": {"type": "array", "items": {
            "type": "object",
            "properties": {"desde": {"type": "string"}, "hasta": {"type": "string"},
                           "etiqueta": {"type": "string"}},
            "required": ["desde", "hasta", "etiqueta"], "additionalProperties": False}},
    },
    "required": ["titulo", "nodos", "conexiones"],
    "additionalProperties": False,
}


def generar_diagrama(motor: Motor, tipo: str, tema: str = "", texto: str = "") -> dict:
    return motor.completar_json(
        f"Crea {TIPOS_DIAGRAMA.get(tipo, tipo)} {_material(tema, texto, motor)}\n"
        "Entre 5 y 15 nodos con textos cortos. Los id son únicos y simples (n1, n2...). "
        "Las conexiones usan esos id; la etiqueta puede ser una cadena vacía.",
        ESQUEMA_DIAGRAMA, SISTEMA)


ESQUEMA_PRESENTACION = {
    "type": "object",
    "properties": {
        "titulo": {"type": "string"},
        "subtitulo": {"type": "string"},
        "diapositivas": {"type": "array", "items": {
            "type": "object",
            "properties": {"titulo": {"type": "string"},
                           "puntos": {"type": "array", "items": {"type": "string"}},
                           "notas": {"type": "string"}},
            "required": ["titulo", "puntos", "notas"], "additionalProperties": False}},
    },
    "required": ["titulo", "subtitulo", "diapositivas"],
    "additionalProperties": False,
}


def generar_presentacion(motor: Motor, tema: str = "", texto: str = "", diapositivas: int = 8) -> dict:
    return motor.completar_json(
        f"Crea una presentación de unas {diapositivas} diapositivas {_material(tema, texto, motor)}\n"
        "Cada diapositiva: un título, 3-5 puntos breves (máx. 15 palabras) y notas del orador "
        "con lo que se diría al presentarla. Incluye una diapositiva final de conclusiones.",
        ESQUEMA_PRESENTACION, SISTEMA)


# ------------------------------------------------------- investigación profunda
ESQUEMA_PLAN = {
    "type": "object",
    "properties": {"subpreguntas": {"type": "array", "items": {"type": "string"}}},
    "required": ["subpreguntas"],
    "additionalProperties": False,
}

SISTEMA_INVESTIGADOR = SISTEMA + """

Ahora actúas como investigador. Busca a fondo (en internet si tienes la herramienta, y en los \
documentos del usuario si existen), contrasta varias fuentes y redacta hallazgos concretos con \
datos, fechas y cifras. Indica la fuente de cada dato. Señala si las fuentes no coinciden."""


def investigar(motor: Motor, pregunta: str, indice: BM25 | None = None, web: bool = True,
               subpreguntas: int = 4, al_progreso: Progreso | None = None) -> dict:
    """Investigación en varios pasos: planificar → investigar cada parte → redactar un informe."""
    avisar = al_progreso or (lambda _m: None)
    avisar("Planificando la investigación…")
    plan = motor.completar_json(
        f"Quiero investigar a fondo: «{pregunta}». Divide la investigación en {subpreguntas} "
        "subpreguntas concretas y complementarias que, respondidas juntas, cubran el tema "
        "(contexto, conceptos clave, datos/evidencias, distintas posturas, aplicaciones o implicaciones).",
        ESQUEMA_PLAN, SISTEMA)
    subs = [s for s in plan.get("subpreguntas", []) if s.strip()][:subpreguntas] or [pregunta]

    usar_web = web and motor.usa_web
    hallazgos, fuentes, documentos = [], [], []
    for n, sub in enumerate(subs, 1):
        avisar(f"Investigando ({n}/{len(subs)}): {sub}")
        texto, encontrados = _con_contexto(sub, indice, k=5)
        documentos += [f"{r.documento}" + (f", pág. {r.pagina}" if r.pagina else "") for r in encontrados]
        r = motor.conversar(SISTEMA_INVESTIGADOR, [{"role": "user", "content":
                            f"Contexto general: {pregunta}\n\nInvestiga esta parte: {texto}"}],
                            Herramientas(indice), web=usar_web, al_progreso=avisar)
        hallazgos.append(f"## {sub}\n\n{r.texto}")
        fuentes += r.fuentes

    avisar("Redactando el informe final…")
    informe = motor.completar(
        f"Pregunta de investigación: «{pregunta}»\n\nHallazgos de cada parte:\n\n" + "\n\n".join(hallazgos) +
        "\n\nRedacta un informe final en Markdown: ## Resumen ejecutivo, secciones temáticas con títulos, "
        "## Puntos de desacuerdo o incertidumbre, ## Conclusiones. Integra los hallazgos sin repetirlos, "
        "conserva las fuentes de los datos y no añadas datos nuevos.", SISTEMA)

    vistas, unicas = set(), []
    for f in fuentes:
        if f["url"] not in vistas:
            vistas.add(f["url"])
            unicas.append(f)
    if unicas or documentos:
        informe += "\n\n## Fuentes\n" + "".join(f"\n- [{f['titulo']}]({f['url']})" for f in unicas)
        informe += "".join(f"\n- 📄 {d}" for d in dict.fromkeys(documentos))
    if web and not motor.usa_web:
        informe += ("\n\n> Nota: este motor no puede buscar en internet; la investigación usó solo sus "
                    "conocimientos y tus documentos. Usa Claude para buscar en la web.")
    return {"informe": informe, "subpreguntas": subs, "hallazgos": hallazgos, "fuentes": unicas}
