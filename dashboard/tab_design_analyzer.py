"""
dashboard/tab_design_analyzer.py — Analizador de Diseños (v2, compatible v5.1)
═══════════════════════════════════════════════════════════════════════════════
Pestaña del dashboard DTF que permite al administrador de marca:
  1. Subir una imagen de diseño (playera, hoodie, etc.)
  2. Ejecutar análisis con OpenAI GPT-4o Vision (keywords + categoría)
  3. Ver la probabilidad de éxito 0-100% con sus 3 componentes desglosados
  4. Recibir recomendación accionable (producir / edición limitada / esperar)

Reescrita para v5.1:
  • Estética fashion con paleta dorado/teal/ámbar en lugar de azul/verde
  • Dual theme-aware — lee COLORS del app.py (dark o light)
  • Tipografía Inter con uppercase tracking en títulos
  • Alertas con <div class="alerta-box"> en vez de st.success/warning
  • Plotly usando CHART_LAYOUT del dashboard (pasado vía colors)
  • Lenguaje de administrador de marca, no jerga técnica

Integración (app.py v5.1):
    from dashboard.tab_design_analyzer import render_design_analyzer_tab
    with tab6:
        render_design_analyzer_tab(API_URL, COLORS)
═══════════════════════════════════════════════════════════════════════════════
"""

import io
import logging
from datetime import datetime
from typing import Optional

import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

log = logging.getLogger("dashboard.design_analyzer")

# Timeouts generosos: Vision API puede tardar 3-8s, Trends puede tardar 5-15s
TIMEOUT_ANALYZE = 45


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS DE RED
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_categorias(api_url: str) -> list:
    """Carga las categorías válidas desde el backend."""
    try:
        r = requests.get(f"{api_url}/api/v1/categorias-disponibles", timeout=5)
        if r.ok:
            return r.json().get("categorias", [])
    except Exception as e:
        log.warning(f"No se pudieron cargar categorías: {e}")
    # Fallback hardcodeado (debe coincidir con services/design_vision.py)
    return [
        "Sports", "Gym", "Futbol", "Basketball", "Tenis", "Casual",
        "Movies", "Musica", "Hockey", "Deportiva", "Skateboarding",
        "Baseball", "Ufc", "Track and Field",
    ]


def _post_analyze(
    api_url: str,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    geo: str,
    categoria_override: Optional[str],
) -> dict:
    """Llama al endpoint POST /api/v1/analyze-design y devuelve el JSON."""
    files = {"file": (filename, io.BytesIO(file_bytes), mime_type)}
    data = {"geo": geo}
    if categoria_override:
        data["categoria_override"] = categoria_override

    r = requests.post(
        f"{api_url}/api/v1/analyze-design",
        files=files,
        data=data,
        timeout=TIMEOUT_ANALYZE,
    )
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"HTTP {r.status_code}: {detail}")
    return r.json()


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS DE ESTILO (heredan paleta del app.py)
# ═══════════════════════════════════════════════════════════════════════════

def _chart_layout(colors: dict) -> dict:
    """Recrea el CHART_LAYOUT del dashboard para mantener coherencia visual."""
    return dict(
        template=colors["_template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=colors["_plot"],
        font=dict(family="sans-serif", color=colors["text"], size=12),
        xaxis=dict(gridcolor=colors["border"], linecolor=colors["border"]),
        yaxis=dict(gridcolor=colors["border"], linecolor=colors["border"]),
    )


def _timestamp_label(colors: dict, prefix: str = "Datos actualizados") -> str:
    """Replica el timestamp_label del app.py (no se puede importar circularmente)."""
    ahora = datetime.now().strftime("%d %b %Y, %H:%M")
    return (
        f'<span style="font-size:0.72rem;letter-spacing:0.08em;'
        f'color:{colors["muted"]};text-transform:uppercase;'
        f'margin-bottom:1.5rem;display:block;">{prefix} · {ahora}</span>'
    )


def _badge(texto: str, tipo: str, colors: dict) -> str:
    """Genera un badge HTML con la estética del dashboard."""
    config = {
        "ok":     (colors["teal_dim"],    colors["teal"]),
        "warn":   (colors["gold_dim"],    colors["amber"]),
        "danger": ("rgba(176,92,92,0.12)", colors["red"]),
        "gold":   (colors["gold_dim"],    colors["gold"]),
    }
    bg, border_color = config.get(tipo, config["gold"])
    return (
        f'<span style="display:inline-block;background:{bg};'
        f'border:1px solid {border_color};color:{border_color};'
        f'font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;'
        f'padding:0.2rem 0.7rem;border-radius:1px;font-weight:500;'
        f'margin:2px 4px 2px 0;">{texto}</span>'
    )


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES PLOTLY (estética fashion)
# ═══════════════════════════════════════════════════════════════════════════

def _render_gauge(score: float, colors: dict) -> go.Figure:
    """
    Velocímetro editorial con paleta fashion: bandas ámbar/dorado/teal
    en lugar de rojo/amarillo/verde tradicionales.
    """
    # Color del indicador según umbral
    if score >= 70:
        bar_color = colors["teal"]       # alto — teal fashion
    elif score >= 40:
        bar_color = colors["gold"]       # medio — dorado
    else:
        bar_color = colors["amber"]      # bajo — ámbar

    # Bandas de fondo adaptadas al tema (dim versions)
    if colors["_template"] == "plotly_dark":
        step_bajo = "rgba(196,147,106,0.18)"
        step_medio = "rgba(194,168,122,0.18)"
        step_alto = "rgba(78,158,138,0.18)"
    else:
        step_bajo = "rgba(160,98,42,0.14)"
        step_medio = "rgba(139,99,16,0.14)"
        step_alto = "rgba(45,122,104,0.14)"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            "suffix": " / 100",
            "font": {"size": 38, "color": colors["text"], "family": "Inter"},
        },
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": colors["muted"],
                "tickfont": {"color": colors["muted"], "size": 11},
            },
            "bar": {"color": bar_color, "thickness": 0.35},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 1,
            "bordercolor": colors["border"],
            "steps": [
                {"range": [0, 40], "color": step_bajo},
                {"range": [40, 70], "color": step_medio},
                {"range": [70, 100], "color": step_alto},
            ],
            "threshold": {
                "line": {"color": colors["text"], "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=colors["text"], family="Inter"),
    )
    return fig


def _render_componentes_chart(componentes: dict, pesos: dict, colors: dict) -> go.Figure:
    """
    Gráfica horizontal mostrando: score bruto de cada componente
    vs. aporte ponderado al score final.
    """
    orden = [
        ("demanda_ml", "Nivel de demanda histórica", pesos["ml"]),
        ("tendencia_trends", "Tendencias de búsqueda", pesos["trends"]),
        ("estacionalidad_hm", "Patrón estacional de la industria", pesos["hm"]),
    ]

    labels, raw_scores, contribuciones = [], [], []
    for key, label, peso in orden:
        comp = componentes.get(key, {})
        raw = comp.get("score", 0)
        labels.append(f"{label}<br><sub style='color:{colors['muted']}'>peso {peso:.0%}</sub>")
        raw_scores.append(raw)
        contribuciones.append(raw * peso)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=raw_scores,
        orientation="h",
        name="Calificación del componente",
        marker_color=colors["gold"],
        opacity=0.85,
        text=[f"{s:.0f}" for s in raw_scores],
        textposition="outside",
        textfont=dict(color=colors["text"], size=11),
    ))
    fig.add_trace(go.Bar(
        y=labels,
        x=contribuciones,
        orientation="h",
        name="Aporte al total",
        marker_color=colors["teal"],
        opacity=0.85,
        text=[f"{c:.0f} pts" for c in contribuciones],
        textposition="outside",
        textfont=dict(color=colors["text"], size=11),
    ))
    fig.update_layout(
        **_chart_layout(colors),
        barmode="group",
        height=360,
        margin=dict(l=20, r=40, t=40, b=60),
        xaxis_title="Puntos",
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.15,
            xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# PESTAÑA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def render_design_analyzer_tab(api_url: str, colors: dict):
    """
    Renderiza la pestaña del analizador de diseños.

    Args:
        api_url: URL del backend FastAPI (ej. http://api:8000)
        colors: diccionario COLORS del dashboard — respeta tema activo
    """
    st.header("Analizador de Diseños")
    st.markdown(_timestamp_label(colors, "Evaluación de diseño"), unsafe_allow_html=True)

    st.caption(
        "Sube un diseño y obtén su índice de prioridad como producto DTF. "
        "Este índice combina tres señales: el historial de tu marca, las "
        "tendencias de búsqueda y los patrones estacionales de la industria. "
        "No es una predicción de ventas, sino una heurística para ordenar "
        "diseños por nivel de atención recomendado."
    )

    # ── Controles de entrada ────────────────────────────────────────────
    col_upload, col_opts = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Imagen del diseño",
            type=["jpg", "jpeg", "png", "webp"],
            help="Máximo 5 MB. Formatos: JPG, PNG, WEBP.",
            key="design_uploader",
        )

    with col_opts:
        categorias = _fetch_categorias(api_url)
        categoria_override = st.selectbox(
            "Categoría",
            options=["Detectar automáticamente"] + categorias,
            index=0,
            help="Opcional: fuerza una categoría en lugar de dejar que la IA la detecte.",
        )
        geo = st.selectbox(
            "Mercado",
            options=["MX", "US", "CO", "AR", "ES"],
            index=0,
            help="País de referencia para medir tendencias de búsqueda.",
        )

    # ── Preview del diseño subido ───────────────────────────────────────
    if uploaded is not None:
        col_preview, col_info = st.columns([1, 2])
        with col_preview:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)
        with col_info:
            size_kb = len(uploaded.getvalue()) / 1024
            st.metric("Archivo", uploaded.name)
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Tamaño", f"{size_kb:.1f} KB")
            with col_m2:
                st.metric("Formato", uploaded.type.split("/")[-1].upper())

    st.divider()

    # ── Botón de análisis ───────────────────────────────────────────────
    analyze = st.button(
        "Evaluar diseño",
        type="primary",
        disabled=(uploaded is None),
        use_container_width=True,
    )

    if not analyze:
        if uploaded is None:
            st.markdown(
                f'<div class="alerta-box">'
                f'<span class="alerta-titulo">Comenzar evaluación</span>'
                f'Sube una imagen del diseño que quieras analizar para obtener '
                f'su probabilidad de éxito en el mercado.'
                f'</div>',
                unsafe_allow_html=True,
            )
        return

    # ── Ejecutar análisis ───────────────────────────────────────────────
    override = None if categoria_override == "Detectar automáticamente" else categoria_override

    with st.spinner("Evaluando diseño... consultando historial, tendencias y patrones de la industria."):
        try:
            resultado = _post_analyze(
                api_url=api_url,
                file_bytes=uploaded.getvalue(),
                filename=uploaded.name,
                mime_type=uploaded.type,
                geo=geo,
                categoria_override=override,
            )
        except requests.exceptions.Timeout:
            st.markdown(
                '<div class="alerta-box alerta-box-danger">'
                '<span class="alerta-titulo alerta-titulo-danger">Evaluacion interrumpida</span>'
                'La evaluación tardó más de 45 segundos. Intenta con una imagen más pequeña '
                'o de menor resolución.'
                '</div>',
                unsafe_allow_html=True,
            )
            return
        except requests.exceptions.ConnectionError:
            st.markdown(
                f'<div class="alerta-box alerta-box-danger">'
                f'<span class="alerta-titulo alerta-titulo-danger">Sin conexion al servidor</span>'
                f'No se pudo conectar al servicio de análisis en <code>{api_url}</code>. '
                f'Verifica que el backend esté corriendo.'
                f'</div>',
                unsafe_allow_html=True,
            )
            return
        except Exception as e:
            st.markdown(
                f'<div class="alerta-box alerta-box-danger">'
                f'<span class="alerta-titulo alerta-titulo-danger">Error en la evaluacion</span>'
                f'{str(e)}'
                f'</div>',
                unsafe_allow_html=True,
            )
            return

    # ═══════════════════════════════════════════════════════════════════════
    # SECCIÓN 1: Veredicto principal (velocímetro + recomendación)
    # ═══════════════════════════════════════════════════════════════════════
    st.subheader("Índice de prioridad")

    score = resultado["score_viabilidad"]
    interp = resultado["interpretacion"]

    col_gauge, col_veredicto = st.columns([1, 1])

    with col_gauge:
        st.plotly_chart(_render_gauge(score, colors), use_container_width=True)

    with col_veredicto:
        nivel = interp["nivel"]

        # Banner adaptado al estilo del dashboard (alerta-box custom)
        if nivel == "ALTA":
            box_class = "alerta-box alerta-box-success"
            titulo_class = "alerta-titulo alerta-titulo-success"
            titulo_txt = "Diseño recomendado"
        elif nivel == "MEDIA":
            box_class = "alerta-box"
            titulo_class = "alerta-titulo"
            titulo_txt = "Viable con precaución"
        else:
            box_class = "alerta-box alerta-box-danger"
            titulo_class = "alerta-titulo alerta-titulo-danger"
            titulo_txt = "No recomendado en este momento"

        # Usar directamente los textos del backend v2 (ya vienen reenmarcados)
        accion_legible = interp["accion_sugerida"]
        recomendacion_legible = interp["recomendacion"]

        st.markdown(
            f'<div class="{box_class}">'
            f'<span class="{titulo_class}">{titulo_txt}</span>'
            f'<p style="margin:0.4rem 0 0.8rem 0;">{recomendacion_legible}</p>'
            f'<p style="margin:0;font-size:0.82rem;color:{colors["muted"]};">{accion_legible}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════
    # SECCIÓN 2: Análisis visual extraído
    # ═══════════════════════════════════════════════════════════════════════
    st.subheader("Lectura del diseño")

    av = resultado["analisis_visual"]

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Categoría detectada", av["categoria_sugerida"])
    col_b.metric("Estilo", av["estilo"].title())

    # Complejidad con color semántico
    comp_nivel = av["complejidad"]
    comp_map = {"baja": "Baja", "media": "Media", "alta": "Alta"}
    col_c.metric(
        "Complejidad de impresión",
        comp_map.get(comp_nivel, comp_nivel),
        help="Baja = costo DTF bajo (pocos colores, formas simples). "
             "Alta = costo DTF alto (degradados, muchos detalles).",
    )

    # Keywords y colores como badges unificados con el resto del dashboard
    col_kw, col_col = st.columns(2)
    with col_kw:
        disclaimer = interp.get("disclaimer", "")

        st.markdown(
            f'<div class="{box_class}">'
            f'<span class="{titulo_class}">{titulo_txt}</span>'
            f'<p style="margin:0.4rem 0 0.8rem 0;">{recomendacion_legible}</p>'
            f'<p style="margin:0 0 0.6rem 0;font-size:0.82rem;color:{colors["muted"]};">{accion_legible}</p>'
            + (
                f'<p style="margin:0;font-size:0.75rem;color:{colors["muted"]};'
                f'font-style:italic;border-top:1px solid {colors["border"]};'
                f'padding-top:0.5rem;">{disclaimer}</p>'
                if disclaimer else ""
            )
            + f'</div>',
            unsafe_allow_html=True,
        )
        chips = "".join([_badge(kw, "gold", colors) for kw in av["keywords"]])
        st.markdown(f'<div style="margin-top:0.6rem;">{chips}</div>', unsafe_allow_html=True)

    with col_col:
        if av.get("colores_dominantes"):
            st.markdown(
                f'<span style="font-size:0.72rem;letter-spacing:0.1em;'
                f'text-transform:uppercase;color:{colors["muted"]};font-weight:500;">'
                f'Colores predominantes</span>',
                unsafe_allow_html=True,
            )
            chips_col = "".join([_badge(c, "ok", colors) for c in av["colores_dominantes"]])
            st.markdown(f'<div style="margin-top:0.6rem;">{chips_col}</div>', unsafe_allow_html=True)

    if av.get("audiencia_estimada"):
        st.markdown(
            f'<div class="alerta-box" style="margin-top:1.2rem;">'
            f'<span class="alerta-titulo">Audiencia probable</span>'
            f'{av["audiencia_estimada"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════
    # SECCIÓN 3: Desglose del score (TRANSPARENCIA ACADÉMICA)
    # ═══════════════════════════════════════════════════════════════════════
    st.subheader("Cómo se calculó este índice")

    st.caption(
        "El índice combina tres señales mediante promedio ponderado. La barra dorada "
        "muestra la calificación de cada señal por separado; la barra teal muestra "
        "cuánto aporta al índice final según su peso. Esta transparencia permite al "
        "usuario entender qué señal impulsa el resultado, algo que un modelo opaco no permite."
    )

    componentes = resultado["componentes_score"]
    #pesos = resultado["pesos_modelo"]
    pesos = resultado["pesos"]                 # ← clave real de la API

    st.plotly_chart(
        _render_componentes_chart(componentes, pesos, colors),
        use_container_width=True,
    )

    # Tabla detallada (oculta por default)
    with st.expander("Ver detalle de cada señal"):
        nombres_humanos = {
            "demanda_ml": "Nivel de demanda histórica de la categoría",
            "tendencia_trends": "Tendencias de búsqueda en Google",
            "estacionalidad_hm": "Patrón estacional de la industria",
        }
        pesos_map = {
            "demanda_ml": "ml",
            "tendencia_trends": "trends",
            "estacionalidad_hm": "hm",
        }

        rows = []
        for key, label in nombres_humanos.items():
            comp = componentes.get(key, {})
            peso = pesos[pesos_map[key]]
            rows.append({
                "Señal evaluada": label,
                "Calificación (0-100)": f"{comp.get('score', 0):.1f}",
                "Peso": f"{peso:.0%}",
                "Aporte al total": f"{comp.get('score', 0) * peso:.1f} pts",
                "Qué significa": comp.get("detalle", ""),
            })

        df_comp = pd.DataFrame(rows)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        st.markdown(
            f'<div class="alerta-box" style="margin-top:0.8rem;">'
            f'<span class="alerta-titulo">Fórmula de cálculo</span>'
            f'Índice final = '
            f'<strong>{pesos["ml"]:.0%}</strong> · demanda histórica '
            f'+ <strong>{pesos["trends"]:.0%}</strong> · tendencia de búsqueda '
            f'+ <strong>{pesos["hm"]:.0%}</strong> · estacionalidad de la industria.'
            f'<br><span style="font-size:0.78rem;color:{colors["muted"]};">'
            f'Se prioriza la señal interna (historial de la marca) sobre las externas '
            f'porque el modelo está validado empíricamente sobre los datos propios '
            f'(MAPE 15.48% en walk-forward backtesting).'
            f'</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Metadata técnica (colapsada, para defensa/debug) ─────────────────
    with st.expander("Datos técnicos de la evaluación"):
        st.json(resultado.get("metadata", {}))