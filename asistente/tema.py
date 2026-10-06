"""Tema visual tipo HUD futurista (azul marino + neón cian) para la app de Streamlit."""

from __future__ import annotations

import html

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&display=swap');

:root {
  --cian: #00d4ff;
  --cian-claro: #7ae8ff;
  --cian-suave: rgba(0, 212, 255, 0.35);
  --cian-tenue: rgba(0, 212, 255, 0.12);
  --fondo: #020b1a;
  --panel: rgba(4, 26, 52, 0.72);
  --texto: #cdf4ff;
  --naranja: #ffb547;
}

html, body, [class*="css"], .stMarkdown, p, li, label, input, textarea, button {
  font-family: 'Rajdhani', 'Segoe UI', sans-serif !important;
  font-size: 1.04rem;
}

/* Fondo: degradado radial + rejilla tenue */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse at 50% 0%, rgba(0, 140, 255, 0.18), transparent 60%),
    radial-gradient(circle at 85% 90%, rgba(0, 212, 255, 0.08), transparent 40%),
    linear-gradient(rgba(0, 212, 255, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 212, 255, 0.035) 1px, transparent 1px),
    var(--fondo);
  background-size: auto, auto, 40px 40px, 40px 40px, auto;
}
[data-testid="stHeader"] { background: transparent; }

h1, h2, h3, h4 {
  font-family: 'Orbitron', sans-serif !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--cian-claro) !important;
  text-shadow: 0 0 10px var(--cian-suave);
}

/* Barra lateral */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(2, 20, 42, 0.97), rgba(1, 10, 24, 0.97));
  border-right: 1px solid var(--cian-suave);
  box-shadow: 4px 0 24px rgba(0, 212, 255, 0.08);
}
[data-testid="stSidebar"] [role="radiogroup"] label {
  border: 1px solid transparent;
  border-radius: 4px;
  padding: 2px 6px;
  transition: all .2s;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  border-color: var(--cian-suave);
  background: var(--cian-tenue);
}

/* Pestañas como botones del HUD */
.stTabs [data-baseweb="tab-list"], [role="tablist"] {
  gap: 6px;
  border-bottom: 1px solid var(--cian-suave);
  flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"], [data-testid="stTab"] {
  font-family: 'Orbitron', sans-serif !important;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--cian-claro);
  background: var(--panel);
  border: 1px solid var(--cian-suave);
  border-bottom: none;
  border-radius: 10px 10px 0 0;
  padding: 8px 14px;
}
.stTabs [aria-selected="true"], [data-testid="stTab"][aria-selected="true"] {
  background: linear-gradient(180deg, rgba(0, 212, 255, 0.28), rgba(0, 212, 255, 0.06));
  box-shadow: 0 0 14px var(--cian-suave), inset 0 0 10px var(--cian-tenue);
  color: #fff !important;
}
.stTabs [data-baseweb="tab-highlight"] { background: var(--cian); box-shadow: 0 0 8px var(--cian); }

/* Botones */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  font-family: 'Orbitron', sans-serif !important;
  font-size: 0.75rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--cian-claro);
  background: rgba(0, 40, 70, 0.6);
  border: 1px solid var(--cian);
  border-radius: 999px;
  box-shadow: 0 0 8px var(--cian-tenue), inset 0 0 8px var(--cian-tenue);
  transition: all .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
  color: #fff;
  border-color: var(--cian-claro);
  box-shadow: 0 0 18px var(--cian-suave), inset 0 0 12px var(--cian-suave);
}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"],
.stDownloadButton > button[kind="primary"] {
  background: linear-gradient(90deg, rgba(0, 150, 220, 0.55), rgba(0, 212, 255, 0.35));
  color: #fff;
}

/* Campos de texto */
input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"] > div {
  background-color: rgba(1, 16, 34, 0.85) !important;
  border-color: var(--cian-suave) !important;
  color: var(--texto) !important;
}
input:focus, textarea:focus { box-shadow: 0 0 10px var(--cian-suave) !important; }

/* Paneles con esquinas de HUD */
[data-testid="stChatMessage"], [data-testid="stExpander"] details,
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stForm"] {
  background: var(--panel);
  border: 1px solid var(--cian-suave) !important;
  border-radius: 6px;
  position: relative;
  box-shadow: inset 0 0 18px rgba(0, 212, 255, 0.06);
}
[data-testid="stChatMessage"]::before, [data-testid="stChatMessage"]::after {
  content: "";
  position: absolute;
  width: 14px; height: 14px;
  border-color: var(--cian);
  border-style: solid;
  filter: drop-shadow(0 0 4px var(--cian));
}
[data-testid="stChatMessage"]::before { top: -1px; left: -1px; border-width: 2px 0 0 2px; }
[data-testid="stChatMessage"]::after { bottom: -1px; right: -1px; border-width: 0 2px 2px 0; }

[data-testid="stChatInput"] > div {
  border: 1px solid var(--cian) !important;
  box-shadow: 0 0 16px var(--cian-suave);
  background: rgba(1, 16, 34, 0.92) !important;
  border-radius: 24px !important;
}

/* Métricas y estados */
[data-testid="stMetricValue"] {
  font-family: 'Orbitron', sans-serif !important;
  color: var(--cian) !important;
  text-shadow: 0 0 12px var(--cian-suave);
}
[data-testid="stStatusWidget"], [data-testid="stStatus"] { border-color: var(--cian-suave) !important; }
code { color: var(--naranja) !important; background: rgba(255, 181, 71, 0.08) !important; }
a { color: var(--cian-claro) !important; }
hr { border-color: var(--cian-suave) !important; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: var(--cian-suave); border-radius: 4px; }
::-webkit-scrollbar-track { background: transparent; }

.block-container { padding-top: 1.2rem; }
</style>
"""


def cabecera_html(motor: str, chats: int, documentos: int, mensajes: int) -> str:
    """Cabecera animada (anillos giratorios, reloj y paneles de estado)."""
    motor = html.escape(motor)
    return f"""
<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600&display=swap" rel="stylesheet">
<style>
  :root {{ --c:#00d4ff; --c2:#7ae8ff; --t:rgba(0,212,255,.35); }}
  * {{ box-sizing:border-box; margin:0; }}
  body {{ background:transparent; color:var(--c2); font-family:'Rajdhani',sans-serif; overflow:hidden; }}
  .hud {{ display:grid; grid-template-columns:1fr 210px 1fr; align-items:center; gap:14px; height:220px; }}
  .lado {{ display:flex; flex-direction:column; gap:10px; }}
  .lado.der {{ align-items:flex-end; }}
  .panel {{ border:1px solid var(--t); background:rgba(4,26,52,.6); padding:8px 14px; min-width:210px;
           clip-path:polygon(0 0,calc(100% - 12px) 0,100% 12px,100% 100%,12px 100%,0 calc(100% - 12px));
           box-shadow:inset 0 0 14px rgba(0,212,255,.08); }}
  .etq {{ font-family:'Orbitron'; font-size:9px; letter-spacing:.18em; color:#5fb8d6; text-transform:uppercase; }}
  .val {{ font-family:'Orbitron'; font-size:20px; color:#fff; text-shadow:0 0 10px var(--c); }}
  .barra {{ height:4px; background:rgba(0,212,255,.15); margin-top:6px; }}
  .barra i {{ display:block; height:100%; background:var(--c); box-shadow:0 0 8px var(--c); animation:carga 3s ease-in-out infinite alternate; }}
  @keyframes carga {{ from {{ width:35% }} to {{ width:92% }} }}
  .titulo {{ font-family:'Orbitron'; font-weight:900; font-size:30px; letter-spacing:.32em; color:#fff;
             text-shadow:0 0 14px var(--c),0 0 30px var(--t); }}
  .sub {{ font-family:'Orbitron'; font-size:9px; letter-spacing:.3em; color:#5fb8d6; margin-top:4px; }}
  .centro {{ position:relative; width:210px; height:210px; }}
  svg {{ position:absolute; inset:0; filter:drop-shadow(0 0 6px var(--c)); }}
  .g1 {{ animation:giro 14s linear infinite; transform-origin:105px 105px; }}
  .g2 {{ animation:giro 9s linear infinite reverse; transform-origin:105px 105px; }}
  .g3 {{ animation:giro 5s linear infinite; transform-origin:105px 105px; }}
  @keyframes giro {{ to {{ transform:rotate(360deg) }} }}
  .nucleo {{ animation:latido 2.4s ease-in-out infinite; transform-origin:105px 105px; }}
  @keyframes latido {{ 50% {{ opacity:.55; transform:scale(.92) }} }}
  .reloj {{ position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; }}
  .hora {{ font-family:'Orbitron'; font-size:22px; color:#fff; text-shadow:0 0 10px var(--c); }}
  .fecha {{ font-size:11px; letter-spacing:.15em; color:#5fb8d6; text-transform:uppercase; }}
  .punto {{ display:inline-block; width:7px; height:7px; border-radius:50%; background:#3dff9e;
            box-shadow:0 0 8px #3dff9e; margin-right:6px; animation:latido 1.6s infinite; }}
</style></head><body>
<div class="hud">
  <div class="lado">
    <div><div class="titulo">J.A.R.V.I.S.</div>
         <div class="sub">ASISTENTE DE ESTUDIO · SISTEMA EN LÍNEA</div></div>
    <div class="panel"><div class="etq"><span class="punto"></span>Núcleo activo</div>
         <div class="val" style="font-size:15px">{motor}</div><div class="barra"><i></i></div></div>
  </div>
  <div class="centro">
    <svg viewBox="0 0 210 210">
      <g class="g1" fill="none" stroke="#00d4ff">
        <circle cx="105" cy="105" r="100" stroke-width="1" stroke-opacity=".5"/>
        <circle cx="105" cy="105" r="94" stroke-width="6" stroke-dasharray="4 6" stroke-opacity=".7"/>
      </g>
      <g class="g2" fill="none" stroke="#7ae8ff">
        <circle cx="105" cy="105" r="80" stroke-width="3" stroke-dasharray="120 40 30 40" />
        <circle cx="105" cy="105" r="72" stroke-width="1" stroke-opacity=".6"/>
      </g>
      <g class="g3" fill="none" stroke="#00d4ff">
        <circle cx="105" cy="105" r="62" stroke-width="8" stroke-dasharray="18 10" stroke-opacity=".55"/>
      </g>
      <g class="nucleo">
        <circle cx="105" cy="105" r="46" fill="rgba(0,212,255,.08)" stroke="#00d4ff" stroke-width="2"/>
      </g>
    </svg>
    <div class="reloj"><div class="hora" id="hora">--:--</div><div class="fecha" id="fecha"></div></div>
  </div>
  <div class="lado der">
    <div class="panel"><div class="etq">Chats</div><div class="val">{chats:02d}</div></div>
    <div class="panel"><div class="etq">Documentos en este chat</div><div class="val">{documentos:02d}</div></div>
    <div class="panel"><div class="etq">Mensajes en este chat</div><div class="val">{mensajes:02d}</div></div>
  </div>
</div>
<script>
  function tic() {{
    const d = new Date();
    document.getElementById('hora').textContent = d.toLocaleTimeString('es', {{hour:'2-digit', minute:'2-digit'}});
    document.getElementById('fecha').textContent = d.toLocaleDateString('es', {{weekday:'short', day:'2-digit', month:'short'}});
  }}
  tic(); setInterval(tic, 1000);
</script></body></html>
"""
