"""Jarvis · Asistente de estudio (interfaz web).

    streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st

from asistente import ciencia, estudio, exportar, tema
from asistente.almacen import Almacen
from asistente.documentos import EXTENSIONES, descargar_url, leer_archivo
from asistente.motores import (ClaudeMotor, JarvisMotor, Motor, MotorNoDisponible, OllamaMotor,
                               bloque_imagen)

st.set_page_config(page_title="Jarvis · Asistente de estudio", page_icon="🤖", layout="wide")

IMAGENES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif",
            "webp": "image/webp"}


@st.cache_resource
def almacen() -> Almacen:
    return Almacen(os.environ.get("JARVIS_DB", "jarvis_datos/jarvis.db"))


@st.cache_resource
def cargar_jarvis(ruta: str) -> JarvisMotor:
    return JarvisMotor(ruta)


def ejecutar(funcion, *args, **kwargs):
    """Llama al motor mostrando los errores de forma amigable."""
    try:
        return funcion(*args, **kwargs)
    except MotorNoDisponible as e:
        st.error(str(e))
    except Exception as e:  # errores de red, de la API, etc.
        nombre = type(e).__name__
        if nombre == "AuthenticationError":
            st.error("La clave de la API de Claude no es válida. Revísala en la barra lateral.")
        elif nombre == "RateLimitError":
            st.error("Demasiadas peticiones seguidas. Espera un momento y vuelve a intentarlo.")
        else:
            st.error(f"Algo salió mal ({nombre}): {e}")
    return None


# -------------------------------------------------------------- barra lateral
def barra_lateral() -> tuple[Motor | None, int, dict]:
    db = almacen()
    with st.sidebar:
        st.title("J.A.R.V.I.S.")
        st.caption("Asistente de estudio · panel de control")

        # ---- chats
        chats = db.listar_chats()
        if not chats:
            db.crear_chat()
            chats = db.listar_chats()
        if st.button("➕ Nuevo chat", use_container_width=True):
            st.session_state.chat_id = db.crear_chat()
            st.rerun()
        ids = [c["id"] for c in chats]
        if st.session_state.get("chat_id") not in ids:
            st.session_state.chat_id = ids[0]
        titulos = {c["id"]: c["titulo"] for c in chats}
        chat_id = st.radio("Chats", ids, format_func=lambda i: titulos[i], key="chat_id",
                           label_visibility="collapsed")
        with st.expander("Opciones del chat"):
            nuevo = st.text_input("Renombrar", value=titulos[chat_id], key=f"renombrar_{chat_id}")
            col1, col2 = st.columns(2)
            if col1.button("Guardar", use_container_width=True) and nuevo.strip():
                db.renombrar_chat(chat_id, nuevo.strip())
                st.rerun()
            if col2.button("🗑️ Borrar", use_container_width=True):
                db.borrar_chat(chat_id)
                del st.session_state["chat_id"]
                st.rerun()

        st.divider()
        # ---- motor
        st.subheader("🧠 Cerebro")
        tipo = st.selectbox("Motor", ["Claude (recomendado)", "Ollama (local, gratis)", "Jarvis (modelo propio)"],
                            label_visibility="collapsed")
        motor = None
        try:
            if tipo.startswith("Claude"):
                clave = st.text_input("Clave de la API de Anthropic", type="password",
                                      value=os.environ.get("ANTHROPIC_API_KEY", ""),
                                      help="Consíguela en https://platform.claude.com/ "
                                           "(o define la variable ANTHROPIC_API_KEY).")
                modelo = st.selectbox("Modelo", ["claude-opus-5-5", "claude-sonnet-5-5"],
                                      help="Opus: el más capaz. Sonnet: más rápido y más barato.")
                esfuerzo = st.select_slider("Profundidad de razonamiento", ["low", "medium", "high", "xhigh"],
                                            value="medium",
                                            help="Más profundidad = mejores respuestas en problemas difíciles, "
                                                 "pero más lento y caro.")
                motor = ClaudeMotor(api_key=clave or None, modelo=modelo, esfuerzo=esfuerzo)
            elif tipo.startswith("Ollama"):
                modelo = st.text_input("Modelo de Ollama", value="llama3.1",
                                       help="Instálalo con: ollama pull llama3.1 (o qwen2.5, gemma2…)")
                url = st.text_input("Dirección", value="http://localhost:11434")
                motor = OllamaMotor(modelo, url)
                st.caption("Sin búsqueda web ni herramientas de cálculo automáticas "
                           "(usa la pestaña 🧮 Calculadora).")
            else:
                ruta = st.text_input("Carpeta del modelo", value="checkpoints/jarvis-chat")
                motor = cargar_jarvis(ruta)
                st.caption("⚠️ Modelo educativo de pocos millones de parámetros: escribe frases "
                           "sencillas, pero no puede resolver tareas reales.")
        except MotorNoDisponible as e:
            st.warning(str(e))
        except Exception as e:
            st.warning(f"No se pudo preparar el motor: {e}")

        st.divider()
        st.subheader("⚙️ Respuestas")
        opciones = {
            "solo_documentos": st.toggle("Responder solo con mis documentos",
                                         help="Jarvis responde únicamente con lo que dicen tus "
                                              "documentos y cita las fuentes."),
            "web": st.toggle("Buscar en internet", value=False,
                             help="Jarvis puede buscar y leer páginas web (solo con Claude).",
                             disabled=not (motor and motor.usa_web)),
        }
    return motor, chat_id, opciones


# ------------------------------------------------------------------ utilidades
def selector_material(db: Almacen, chat_id: int, clave: str) -> tuple[str, str, str]:
    """Elige de dónde sale el contenido: documentos, un tema o texto pegado.
    Devuelve (tema, texto, título)."""
    docs = db.documentos(chat_id)
    opciones = (["Mis documentos"] if docs else []) + ["Un tema", "Texto pegado"]
    fuente = st.radio("Material", opciones, horizontal=True, key=f"fuente_{clave}")
    if fuente == "Mis documentos":
        nombres = {d["id"]: d["nombre"] for d in docs}
        elegidos = st.multiselect("Documentos", list(nombres), default=list(nombres)[:1],
                                  format_func=nombres.get, key=f"docs_{clave}")
        texto = "\n\n".join(f"# {nombres[i]}\n{db.texto_documento(i)}" for i in elegidos)
        enfoque = st.text_input("Centrarse en (opcional)", key=f"enfoque_{clave}",
                                placeholder="p. ej. el capítulo 3, las leyes de Newton…")
        titulo = ", ".join(nombres[i] for i in elegidos) or "los documentos"
        return enfoque, texto, titulo
    if fuente == "Un tema":
        tema = st.text_input("Tema", key=f"tema_{clave}", placeholder="p. ej. la mitosis, derivadas…")
        return tema, "", tema
    texto = st.text_area("Pega aquí el texto", height=200, key=f"texto_{clave}")
    return "", texto, "el texto"


def descargas(nombre: str, markdown: str, clave: str) -> None:
    col1, col2 = st.columns(2)
    col1.download_button("⬇️ Markdown", markdown.encode("utf-8"), f"{nombre}.md", "text/markdown",
                         key=f"md_{clave}", use_container_width=True)
    col2.download_button("⬇️ Word", exportar.markdown_a_docx(markdown, nombre), f"{nombre}.docx",
                         key=f"docx_{clave}", use_container_width=True)


def estado(clave: str, chat_id: int):
    return st.session_state.setdefault(f"{clave}_{chat_id}", {})


# ----------------------------------------------------------------------- chat
def pestana_chat(motor: Motor | None, db: Almacen, chat_id: int, opciones: dict) -> None:
    mensajes = db.mensajes(chat_id)
    if not mensajes:
        st.markdown("### 👋 ¡Hola! Soy Jarvis.\n"
                    "Puedo resolver problemas de matemáticas, física y química paso a paso, explicarte "
                    "conceptos, responder según tus apuntes y PDFs, y buscar en internet. "
                    "Adjunta PDFs o fotos de ejercicios con el clip 📎.")
    for m in mensajes:
        with st.chat_message(m["rol"], avatar="🤖" if m["rol"] == "assistant" else "🧑‍🎓"):
            st.markdown(m["texto"])
            mostrar_extras(m["extra"])

    entrada = st.chat_input("Pregúntale a Jarvis…", accept_file="multiple",
                            file_type=EXTENSIONES + list(IMAGENES))
    if not entrada:
        return
    if motor is None:
        st.error("Configura un motor en la barra lateral.")
        return
    texto = (entrada.text or "").strip()
    imagenes = []
    for archivo in entrada.files:
        ext = archivo.name.rsplit(".", 1)[-1].lower()
        if ext in IMAGENES:
            imagenes.append(bloque_imagen(archivo.getvalue(), IMAGENES[ext]))
        else:
            agregar_archivo(db, chat_id, archivo.name, archivo.getvalue())
    if not texto:
        if not imagenes:
            st.rerun()
        texto = "Resuelve o explica lo que aparece en la imagen."
    if imagenes and isinstance(motor, JarvisMotor):
        st.warning("El modelo propio no puede ver imágenes; se ignorarán.")

    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(texto)
    with st.chat_message("assistant", avatar="🤖"):
        with st.status("Pensando…") as caja:
            respuesta = ejecutar(estudio.responder, motor, db, chat_id, texto,
                                 solo_documentos=opciones["solo_documentos"], web=opciones["web"],
                                 imagenes=imagenes, al_progreso=lambda m: caja.update(label=m))
            caja.update(label="Listo", state="complete")
    if respuesta:
        st.rerun()


def mostrar_extras(extra: dict) -> None:
    if extra.get("fuentes"):
        with st.expander(f"🌐 Fuentes web ({len(extra['fuentes'])})"):
            for f in extra["fuentes"]:
                st.markdown(f"- [{f['titulo']}]({f['url']})")
    if extra.get("documentos"):
        with st.expander(f"📄 Fragmentos de tus documentos ({len(extra['documentos'])})"):
            for n, d in enumerate(extra["documentos"], 1):
                donde = f"{d['documento']}, pág. {d['pagina']}" if d.get("pagina") else d["documento"]
                st.markdown(f"**[{n}] {donde}**")
                st.caption(d["texto"][:700] + ("…" if len(d["texto"]) > 700 else ""))
    if extra.get("herramientas"):
        with st.expander(f"🧮 Herramientas usadas ({len(extra['herramientas'])})"):
            for h in extra["herramientas"]:
                st.markdown(f"**{h['nombre']}** `{h['entrada']}`")
                st.code(h["resultado"][:1500], language=None)


# ----------------------------------------------------------------- documentos
def agregar_archivo(db: Almacen, chat_id: int, nombre: str, datos: bytes) -> bool:
    try:
        paginas = leer_archivo(nombre, datos)
        db.agregar_documento(chat_id, nombre, paginas)
        st.toast(f"📄 {nombre} añadido")
        return True
    except Exception as e:
        st.error(f"No se pudo leer {nombre}: {e}")
        return False


def pestana_documentos(db: Almacen, chat_id: int) -> None:
    st.markdown("Los documentos de este chat se usan para responder tus preguntas, hacer resúmenes, "
                "quizzes y diagramas. Jarvis busca en ellos los fragmentos relevantes y cita la página.")
    col1, col2 = st.columns(2)
    with col1:
        with st.form("subir", clear_on_submit=True):
            archivos = st.file_uploader("Sube PDFs, Word, texto o HTML", type=EXTENSIONES,
                                        accept_multiple_files=True)
            if st.form_submit_button("Añadir archivos") and archivos:
                for a in archivos:
                    agregar_archivo(db, chat_id, a.name, a.getvalue())
    with col2:
        with st.form("url", clear_on_submit=True):
            url = st.text_input("…o importa una página web o un PDF en línea",
                                placeholder="https://es.wikipedia.org/wiki/Fotosíntesis")
            if st.form_submit_button("Importar") and url.strip():
                with st.spinner("Descargando…"):
                    try:
                        titulo, paginas = descargar_url(url.strip())
                        db.agregar_documento(chat_id, titulo, paginas, origen=url.strip())
                        st.success(f"🌐 {titulo} añadido")
                    except Exception as e:
                        st.error(f"No se pudo importar: {e}")

    docs = db.documentos(chat_id)
    st.subheader(f"📚 Documentos de este chat ({len(docs)})")
    if not docs:
        st.info("Aún no hay documentos.")
    for d in docs:
        icono = "🌐" if d["origen"].startswith("http") else "📄"
        paginas = f" · {d['paginas']} págs." if d["paginas"] else ""
        with st.expander(f"{icono} {d['nombre']} — {d['caracteres']:,} caracteres{paginas}"):
            if d["origen"].startswith("http"):
                st.markdown(f"Origen: {d['origen']}")
            texto = db.texto_documento(d["id"])
            st.text(texto[:3000] + ("\n…" if len(texto) > 3000 else ""))
            if st.button("Quitar documento", key=f"quitar_{d['id']}"):
                db.borrar_documento(d["id"])
                st.rerun()


# ------------------------------------------------------------------ resúmenes
def pestana_resumenes(motor: Motor | None, db: Almacen, chat_id: int) -> None:
    tema, texto, titulo = selector_material(db, chat_id, "resumen")
    formato = st.selectbox("Formato", list(estudio.FORMATOS_RESUMEN))
    st.caption(estudio.FORMATOS_RESUMEN[formato])
    instrucciones = st.text_input("Indicaciones extra (opcional)",
                                  placeholder="p. ej. máximo una página, enfocado al examen…")
    res = estado("resumen", chat_id)
    if st.button("✨ Crear resumen", type="primary", disabled=motor is None):
        if not texto and not tema:
            st.warning("Elige un documento, escribe un tema o pega un texto.")
        else:
            if texto and tema:  # documentos + enfoque
                instrucciones = f"Céntrate en: {tema}. {instrucciones}"
            material = texto or ejecutar(
                motor.completar, f"Escribe unos apuntes de clase completos y rigurosos sobre «{tema}».",
                estudio.SISTEMA)
            if material:
                with st.status("Resumiendo…") as caja:
                    salida = ejecutar(estudio.resumir, motor, material, formato, titulo, instrucciones,
                                      al_progreso=lambda m: caja.update(label=m))
                    caja.update(label="Listo", state="complete")
                if salida:
                    res.update(texto=salida, nombre=f"{formato} - {titulo}"[:80])
    if res.get("texto"):
        st.divider()
        st.markdown(res["texto"], unsafe_allow_html=True)
        descargas(res["nombre"], res["texto"], "resumen")


# ------------------------------------------------------------------- explicar
def pestana_explicar(motor: Motor | None, db: Almacen, chat_id: int) -> None:
    concepto = st.text_input("¿Qué concepto quieres entender?",
                             placeholder="p. ej. la entropía, la regla de la cadena, el enlace covalente…")
    nivel = st.segmented_control("Nivel", list(estudio.NIVELES), default="Secundaria")
    usar_docs = st.checkbox("Apoyarse en mis documentos", value=bool(db.documentos(chat_id)))
    res = estado("explicar", chat_id)
    if st.button("💡 Explicar", type="primary", disabled=motor is None or not concepto.strip()):
        with st.status("Preparando la explicación…") as caja:
            r = ejecutar(estudio.explicar, motor, concepto, nivel or "Secundaria",
                         db.indice(chat_id) if usar_docs else None,
                         al_progreso=lambda m: caja.update(label=m))
            caja.update(label="Listo", state="complete")
        if r:
            res.update(texto=r.texto, nombre=f"Explicación - {concepto}"[:80])
    if res.get("texto"):
        st.divider()
        st.markdown(res["texto"], unsafe_allow_html=True)
        descargas(res["nombre"], res["texto"], "explicar")


# ---------------------------------------------------------------------- quiz
def pestana_quiz(motor: Motor | None, db: Almacen, chat_id: int) -> None:
    modo = st.segmented_control("Actividad", ["❓ Quiz", "🃏 Tarjetas de estudio"], default="❓ Quiz")
    tema, texto, titulo = selector_material(db, chat_id, "quiz")
    if not (tema or texto):
        st.info("Elige un documento, escribe un tema o pega un texto.")

    if modo == "🃏 Tarjetas de estudio":
        cantidad = st.slider("Número de tarjetas", 5, 40, 15)
        res = estado("tarjetas", chat_id)
        if st.button("🃏 Crear tarjetas", type="primary", disabled=motor is None or not (tema or texto)):
            with st.spinner("Creando tarjetas…"):
                tarjetas = ejecutar(estudio.generar_tarjetas, motor, tema, texto, cantidad)
            if tarjetas:
                res.update(tarjetas=tarjetas, i=0, girada=False)
        if res.get("tarjetas"):
            mostrar_tarjetas(res)
        return

    col1, col2 = st.columns(2)
    cantidad = col1.slider("Número de preguntas", 3, 25, 8)
    dificultad = col2.select_slider("Dificultad", ["fácil", "media", "difícil", "de examen"], value="media")
    tipos = st.multiselect("Tipos de pregunta", list(estudio.TIPOS_PREGUNTA),
                           default=["Opción múltiple", "Verdadero o falso"])
    res = estado("quiz", chat_id)
    if st.button("❓ Crear quiz", type="primary", disabled=motor is None or not (tema or texto) or not tipos):
        with st.spinner("Creando el quiz…"):
            quiz = ejecutar(estudio.generar_quiz, motor, tema, texto, cantidad, dificultad, tipos)
        if quiz:
            res.clear()
            res.update(quiz=quiz, ronda=res.get("ronda", 0) + 1)
    if res.get("quiz"):
        mostrar_quiz(motor, res)


def mostrar_quiz(motor: Motor, res: dict) -> None:
    quiz, ronda = res["quiz"], res["ronda"]
    st.subheader(quiz.get("titulo", "Quiz"))
    with st.form(f"quiz_{ronda}"):
        respuestas = []
        for n, p in enumerate(quiz["preguntas"]):
            st.markdown(f"**{n + 1}. {p['pregunta']}**")
            if p["tipo"] == "respuesta_corta" or not p["opciones"]:
                respuestas.append(st.text_input("Tu respuesta", key=f"q{ronda}_{n}", label_visibility="collapsed"))
            else:
                respuestas.append(st.radio("Opciones", p["opciones"], index=None, key=f"q{ronda}_{n}",
                                           label_visibility="collapsed"))
        enviado = st.form_submit_button("✅ Corregir", type="primary")
    if enviado:
        resultados = []
        with st.spinner("Corrigiendo…"):
            for p, r in zip(quiz["preguntas"], respuestas):
                if not r:
                    resultados.append((False, "Sin responder."))
                elif p["tipo"] == "respuesta_corta" or not p["opciones"]:
                    c = ejecutar(estudio.corregir_respuesta, motor, p["pregunta"], p["respuesta_correcta"], r)
                    resultados.append((bool(c and c["correcta"]), c["comentario"] if c else ""))
                else:
                    resultados.append((r.strip().lower() == p["respuesta_correcta"].strip().lower(), ""))
        res["resultados"] = resultados
    if res.get("resultados"):
        aciertos = sum(ok for ok, _ in res["resultados"])
        total = len(res["resultados"])
        nota = 10 * aciertos / total if total else 0
        st.metric("Resultado", f"{aciertos}/{total}", f"Nota: {nota:.1f}/10")
        if nota >= 9:
            st.balloons()
        for n, (p, (ok, comentario)) in enumerate(zip(quiz["preguntas"], res["resultados"]), 1):
            with st.expander(f"{'✅' if ok else '❌'} {n}. {p['pregunta']}"):
                st.markdown(f"**Respuesta correcta:** {p['respuesta_correcta']}")
                if comentario:
                    st.markdown(f"**Comentario:** {comentario}")
                st.markdown(p["explicacion"])
    st.download_button("⬇️ Descargar quiz (Word)", exportar.markdown_a_docx(exportar.quiz_a_markdown(quiz)),
                       "quiz.docx")


def mostrar_tarjetas(res: dict) -> None:
    tarjetas, i = res["tarjetas"], res["i"] % len(res["tarjetas"])
    t = tarjetas[i]
    st.caption(f"Tarjeta {i + 1} de {len(tarjetas)}")
    with st.container(border=True):
        st.markdown(f"### {t['reverso'] if res['girada'] else t['frente']}")
        st.caption("Reverso" if res["girada"] else "Frente")
    col1, col2, col3 = st.columns(3)
    if col1.button("⬅️ Anterior", use_container_width=True):
        res.update(i=i - 1, girada=False)
        st.rerun()
    if col2.button("🔄 Girar", use_container_width=True, type="primary"):
        res["girada"] = not res["girada"]
        st.rerun()
    if col3.button("Siguiente ➡️", use_container_width=True):
        res.update(i=i + 1, girada=False)
        st.rerun()
    st.download_button("⬇️ Exportar para Anki / Quizlet (CSV)", exportar.tarjetas_a_csv(tarjetas),
                       "tarjetas.csv", "text/csv")


# ------------------------------------------------------------------ diagramas
def pestana_diagramas(motor: Motor | None, db: Almacen, chat_id: int) -> None:
    tipo = st.segmented_control("Tipo", ["🧠 Mapa mental", "🔀 Diagrama", "📊 Presentación (PowerPoint)"],
                                default="🧠 Mapa mental")
    tema, texto, titulo = selector_material(db, chat_id, "diagrama")
    res = estado("diagrama", chat_id)
    listo = motor is not None and bool(tema or texto)

    if tipo == "🧠 Mapa mental":
        if st.button("🧠 Crear mapa mental", type="primary", disabled=not listo):
            with st.spinner("Organizando las ideas…"):
                mapa = ejecutar(estudio.generar_mapa_mental, motor, tema, texto)
            if mapa:
                res.clear()
                res.update(mapa=mapa)
        if res.get("mapa"):
            mapa = res["mapa"]
            st.graphviz_chart(exportar.mapa_a_dot(mapa), use_container_width=True)
            mermaid = exportar.mapa_a_mermaid(mapa)
            col1, col2, col3 = st.columns(3)
            col1.download_button("⬇️ Página web (HTML)", exportar.mermaid_a_html(mermaid, mapa["titulo"]),
                                 "mapa_mental.html", "text/html", use_container_width=True)
            col2.download_button("⬇️ Graphviz (.dot)", exportar.mapa_a_dot(mapa), "mapa_mental.dot",
                                 use_container_width=True)
            col3.download_button("⬇️ Esquema (Word)",
                                 exportar.markdown_a_docx(exportar.mapa_a_markdown(mapa)),
                                 "mapa_mental.docx", use_container_width=True)
            with st.expander("Ver como esquema"):
                st.markdown(exportar.mapa_a_markdown(mapa))

    elif tipo == "🔀 Diagrama":
        clase = st.selectbox("Clase de diagrama", list(estudio.TIPOS_DIAGRAMA))
        if st.button("🔀 Crear diagrama", type="primary", disabled=not listo):
            with st.spinner("Dibujando…"):
                diagrama = ejecutar(estudio.generar_diagrama, motor, clase, tema, texto)
            if diagrama:
                res.clear()
                res.update(diagrama=diagrama, clase=clase)
        if res.get("diagrama"):
            d, clase = res["diagrama"], res["clase"]
            st.markdown(f"#### {d['titulo']}")
            dot = exportar.diagrama_a_dot(d, clase)
            st.graphviz_chart(dot, use_container_width=True)
            mermaid = exportar.diagrama_a_mermaid(d, clase)
            col1, col2 = st.columns(2)
            col1.download_button("⬇️ Página web (HTML)", exportar.mermaid_a_html(mermaid, d["titulo"]),
                                 "diagrama.html", "text/html", use_container_width=True)
            col2.download_button("⬇️ Graphviz (.dot)", dot, "diagrama.dot", use_container_width=True)

    else:
        n = st.slider("Número de diapositivas", 4, 20, 8)
        if st.button("📊 Crear presentación", type="primary", disabled=not listo):
            with st.spinner("Preparando las diapositivas…"):
                pres = ejecutar(estudio.generar_presentacion, motor, tema, texto, n)
            if pres:
                res.clear()
                res.update(pres=pres)
        if res.get("pres"):
            pres = res["pres"]
            st.download_button("⬇️ Descargar PowerPoint (.pptx)", exportar.presentacion_a_pptx(pres),
                               "presentacion.pptx", type="primary")
            st.markdown(exportar.presentacion_a_markdown(pres))


# ------------------------------------------------------- investigación profunda
def pestana_investigar(motor: Motor | None, db: Almacen, chat_id: int, opciones: dict) -> None:
    st.markdown("Jarvis divide tu pregunta en partes, investiga cada una (en internet y en tus "
                "documentos), contrasta las fuentes y redacta un informe con referencias.")
    pregunta = st.text_area("¿Qué quieres investigar?", height=90,
                            placeholder="p. ej. ¿Cómo funcionan las baterías de estado sólido y "
                                        "qué ventajas tienen frente a las de litio?")
    col1, col2 = st.columns(2)
    partes = col1.slider("Profundidad (subpreguntas)", 2, 8, 4)
    web = col2.toggle("Buscar en internet", value=bool(motor and motor.usa_web),
                      disabled=not (motor and motor.usa_web), key="web_investigar")
    if motor and not motor.usa_web:
        st.caption("Este motor no puede buscar en internet: se usarán sus conocimientos y tus documentos.")
    res = estado("investigar", chat_id)
    if st.button("🔎 Investigar", type="primary", disabled=motor is None or not pregunta.strip()):
        with st.status("Investigando… (puede tardar unos minutos)", expanded=True) as caja:
            r = ejecutar(estudio.investigar, motor, pregunta, db.indice(chat_id), web, partes,
                         al_progreso=lambda m: (caja.write(m), caja.update(label=m)))
            caja.update(label="Investigación terminada", state="complete", expanded=False)
        if r:
            res.update(r, nombre=f"Investigación - {pregunta[:60]}")
    if res.get("informe"):
        st.divider()
        st.markdown(res["informe"])
        descargas(res["nombre"], res["informe"], "investigar")
        with st.expander("Ver los hallazgos de cada parte"):
            for h in res["hallazgos"]:
                st.markdown(h)


# ------------------------------------------------------------------ calculadora
def pestana_calculadora() -> None:
    st.markdown("Cálculo exacto (sin inteligencia artificial): funciona siempre, con cualquier motor.")
    tipo = st.segmented_control("Herramienta", ["Matemáticas", "Masa molar", "Ajustar reacción"],
                                default="Matemáticas")
    try:
        if tipo == "Matemáticas":
            col1, col2 = st.columns([1, 3])
            op = col1.selectbox("Operación", ciencia.OPERACIONES, index=4)
            expr = col2.text_input("Expresión", value="x^2 - 5x + 6 = 0",
                                   help="Usa x^2, sqrt(x), sin(x), exp(x), log(x), pi. "
                                        "Varias ecuaciones: x + y = 3, x - y = 1. Matriz: [[1,2],[3,4]]")
            c1, c2, c3, c4 = st.columns(4)
            var = c1.text_input("Variable", placeholder="x")
            desde = c2.text_input("Desde", help="Integral definida / punto de la serie")
            hasta = c3.text_input("Hasta", help="Integral definida / punto del límite")
            orden = c4.number_input("Orden", 1, 20, 1)
            if st.button("Calcular", type="primary"):
                r = ciencia.calcular(op, expr, var, desde, hasta, int(orden))
                if r.get("latex"):
                    st.latex(r["latex"])
                st.json(r)
        elif tipo == "Masa molar":
            formula = st.text_input("Fórmula", value="H2SO4")
            if st.button("Calcular", type="primary"):
                r = ciencia.masa_molar(formula)
                st.metric("Masa molar", f"{r['masa_molar_g_mol']} g/mol")
                st.table(r["detalle"])
        else:
            ec = st.text_input("Reacción", value="C3H8 + O2 -> CO2 + H2O")
            if st.button("Ajustar", type="primary"):
                r = ciencia.balancear_ecuacion(ec)
                st.success(r["ecuacion_ajustada"])
    except Exception as e:
        st.error(f"Error: {e}")


# ----------------------------------------------------------------------- main
def main() -> None:
    st.markdown(tema.CSS, unsafe_allow_html=True)
    motor, chat_id, opciones = barra_lateral()
    db = almacen()
    st.iframe(tema.cabecera_html(motor.nombre if motor else "Sin conectar", len(db.listar_chats()),
                                       len(db.documentos(chat_id)), len(db.mensajes(chat_id))), height=230)
    tabs = st.tabs(["💬 Chat", "📚 Documentos", "📝 Resúmenes", "💡 Explicar", "❓ Quiz y tarjetas",
                    "🧠 Diagramas", "🔎 Investigación profunda", "🧮 Calculadora"])
    with tabs[0]:
        pestana_chat(motor, db, chat_id, opciones)
    with tabs[1]:
        pestana_documentos(db, chat_id)
    with tabs[2]:
        pestana_resumenes(motor, db, chat_id)
    with tabs[3]:
        pestana_explicar(motor, db, chat_id)
    with tabs[4]:
        pestana_quiz(motor, db, chat_id)
    with tabs[5]:
        pestana_diagramas(motor, db, chat_id)
    with tabs[6]:
        pestana_investigar(motor, db, chat_id, opciones)
    with tabs[7]:
        pestana_calculadora()


main()
