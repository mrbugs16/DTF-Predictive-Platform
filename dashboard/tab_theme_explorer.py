"""
dashboard/tab_theme_explorer.py — Explorador de Temas
═══════════════════════════════════════════════════════════════════════════════
Nueva pestaña (tab 7 en app.py v5.2) que permite al administrador de marca:

  1. Ingresar un tema o concepto (ej. "Copa del Mundo", "Halloween")
  2. Ver gráfica de Google Trends del tema en México (últimos 12 meses)
  3. Ver predicción real del modelo ML ganador para los próximos 30 días
  4. Recibir recomendación cuantitativa en piezas (no solo score)

Diferencia fundamental vs Analizador de Diseños:
  • Analizador: input imagen → score heurístico 0-100
  • Explorador:  input texto → predicción cuantitativa en piezas usando ML

Integración en app.py v5.2:
    from dashboard.tab_theme_explorer import render_theme_explorer_tab
    with tab7:
        render_theme_explorer_tab(API_URL, COLORS)
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

log = logging.getLogger("dashboard.theme_explorer")

TIMEOUT_ANALYZE = 45


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _chart_layout(colors: dict) -> dict:
    return dict(
        template=colors["_template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=colors["_plot"],
        font=dict(family="sans-serif", color=colors["text"], size=12),
        xaxis=dict(gridcolor=colors["border"], linecolor=colors["border"]),
        yaxis=dict(gridcolor=colors["border"], linecolor=colors["border"]),
    )


def _timestamp_label(colors: dict, prefix: str = "Análisis") -> str:
    ahora = datetime.now().strftime("%d %b %Y, %H:%M")
    return (
        f'<span style="font-size:0.72rem;letter-spacing:0.08em;'
        f'color:{colors["muted"]};text-transform:uppercase;'
        f'margin-bottom:1.5rem;display:block;">{prefix} · {ahora}</span>'
    )


def _post_analyze_theme(
    api_url: str,
    tema: str,
    geo: str,
    categoria_override: Optional[str],
    horizonte_dias: int,
) -> dict:
    """Llama al endpoint POST /api/v1/analyze-theme."""
    body = {
        "tema": tema,
        "geo": geo,
        "horizonte_dias": horizonte_dias,
    }
    if categoria_override:
        body["categoria_override"] = categoria_override

    r = requests.post(
        f"{api_url}/api/v1/analyze-theme",
        json=body,
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
# VISUALIZACIONES
# ═══════════════════════════════════════════════════════════════════════════

def _render_trends_chart(serie_trends: list, tema: str, colors: dict) -> go.Figure:
    """Gráfica de línea del interés de Google Trends en el tiempo."""
    if not serie_trends:
        return None

    df = pd.DataFrame(serie_trends)
    df["fecha"] = pd.to_datetime(df["fecha"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["fecha"],
        y=df["interes"],
        mode="lines",
        name=tema,
        line=dict(color=colors["gold"], width=2.5),
        fill="tozeroy",
        fillcolor=colors["gold_dim"],
    ))

    # Línea de promedio
    promedio = df["interes"].mean()
    fig.add_hline(
        y=promedio,
        line_dash="dash",
        line_color=colors["muted"],
        annotation_text=f"Promedio: {promedio:.0f}",
        annotation_font_color=colors["muted"],
    )

    fig.update_layout(
        **_chart_layout(colors),
        title=f"Interés en '{tema}' — últimos 12 meses",
        yaxis_title="Interés relativo (0-100)",
        xaxis_title=None,
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def _render_forecast_chart(forecast: dict, recomendacion: dict, colors: dict) -> go.Figure:
    """
    Gráfica de barras horizontal mostrando la recomendación cuantitativa.
    Banda inferior — central — banda superior.
    """
    rango = recomendacion.get("piezas_estimadas_rango", {})
    central = recomendacion.get("piezas_estimadas_central", 0)
    minimo = rango.get("minimo", 0)
    maximo = rango.get("maximo", 0)

    fig = go.Figure()

    # Barra de rango (invisible, para crear el efecto visual)
    fig.add_trace(go.Bar(
        y=["Piezas estimadas"],
        x=[maximo - minimo],
        base=[minimo],
        orientation="h",
        marker_color=colors["gold_dim"],
        marker_line_color=colors["gold"],
        marker_line_width=1,
        name="Rango estimado",
        text=f"{minimo:.0f} – {maximo:.0f}",
        textposition="outside",
        textfont=dict(color=colors["text"]),
        hovertemplate=f"Mínimo: {minimo:.0f}<br>Máximo: {maximo:.0f}<extra></extra>",
    ))

    # Marcador para el valor central
    fig.add_trace(go.Scatter(
        y=["Piezas estimadas"],
        x=[central],
        mode="markers+text",
        marker=dict(
            symbol="diamond",
            size=20,
            color=colors["teal"],
            line=dict(width=2, color=colors["text"]),
        ),
        text=[f"{central:.0f}"],
        textposition="top center",
        textfont=dict(color=colors["teal"], size=14, family="Inter"),
        name="Estimación central",
        hovertemplate=f"Central: {central:.0f}<extra></extra>",
    ))

    fig.update_layout(
        **_chart_layout(colors),
        title=f"Proyección en piezas — {forecast['horizonte_dias']} días",
        xaxis_title="Piezas",
        height=200,
        margin=dict(l=20, r=80, t=50, b=20),
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# PESTAÑA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def render_theme_explorer_tab(api_url: str, colors: dict):
    """Renderiza la pestaña del Explorador de Temas."""

    st.header("Explorador de Temas")
    st.markdown(_timestamp_label(colors, "Análisis de oportunidad"),
                unsafe_allow_html=True)

    st.caption(
        "Ingresa un tema, evento o concepto (ej. 'Copa del Mundo 2026', "
        "'Halloween', 'Día de Muertos') y obtén una estimación cuantitativa "
        "de cuántas piezas podrías vender según los modelos entrenados, "
        "modulada por el interés actual en Google."
    )

    # ── Inputs ───────────────────────────────────────────────────────────
    col_tema, col_geo, col_horizonte = st.columns([3, 1, 1])

    with col_tema:
        tema = st.text_input(
            "Tema o concepto a explorar",
            placeholder="Ej: Copa del Mundo 2026",
            help="Texto libre. Intenta ser específico pero no demasiado largo.",
        )

    with col_geo:
        geo = st.selectbox(
            "Mercado",
            options=["MX", "US", "CO", "AR", "ES"],
            index=0,
        )

    with col_horizonte:
        horizonte_dias = st.selectbox(
            "Horizonte",
            options=[15, 30, 60],
            index=1,
            format_func=lambda x: f"{x} días",
        )

    # Categoría opcional
    categorias = [
        "Sports", "Gym", "Futbol", "Basketball", "Tenis", "Casual",
        "Movies", "Musica", "Hockey", "Deportiva", "Skateboarding",
        "Baseball", "Ufc", "Track and Field",
    ]
    categoria_override = st.selectbox(
        "Categoría DTF (opcional)",
        options=["Detectar automáticamente"] + categorias,
        index=0,
        help="El sistema detecta la categoría por palabras clave. "
             "Usa esta opción si quieres forzar una categoría específica.",
    )

    # ── Botón de análisis ───────────────────────────────────────────────
    analyze = st.button(
        "Explorar tema",
        type="primary",
        disabled=(not tema or len(tema.strip()) < 2),
        use_container_width=True,
    )

    if not analyze:
        if not tema:
            st.markdown(
                f'<div class="alerta-box">'
                f'<span class="alerta-titulo">Comenzar exploración</span>'
                f'Escribe un tema, evento o concepto para analizar su potencial. '
                f'El sistema consultará Google Trends y los modelos entrenados para '
                f'estimar cuántas piezas se venderían si lanzaras merch sobre ese tema.'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Sugerencias de ejemplo
            st.markdown(
                f'<span style="font-size:0.72rem;letter-spacing:0.1em;'
                f'text-transform:uppercase;color:{colors["muted"]};">'
                f'Ejemplos para probar</span>',
                unsafe_allow_html=True,
            )
            ejemplos_html = "".join([
                f'<span style="display:inline-block;background:{colors["gold_dim"]};'
                f'border:1px solid {colors["gold"]};color:{colors["gold"]};'
                f'font-size:0.78rem;padding:0.3rem 0.8rem;border-radius:2px;'
                f'margin:4px 6px 4px 0;">{ej}</span>'
                for ej in [
                    "Copa del Mundo 2026",
                    "Halloween",
                    "Día de Muertos",
                    "Navidad",
                    "Mundial de Futbol",
                    "Super Bowl",
                    "Año Nuevo",
                ]
            ])
            st.markdown(f'<div style="margin-top:0.6rem;">{ejemplos_html}</div>',
                        unsafe_allow_html=True)
        return

    # ── Ejecutar análisis ───────────────────────────────────────────────
    override = None if categoria_override == "Detectar automáticamente" else categoria_override

    with st.spinner("Consultando Google Trends y modelos entrenados..."):
        try:
            resultado = _post_analyze_theme(
                api_url=api_url,
                tema=tema.strip(),
                geo=geo,
                categoria_override=override,
                horizonte_dias=horizonte_dias,
            )
        except requests.exceptions.Timeout:
            st.markdown(
                '<div class="alerta-box alerta-box-danger">'
                '<span class="alerta-titulo alerta-titulo-danger">Tiempo agotado</span>'
                'El análisis tardó más de 45 segundos. Intenta con un tema más específico.'
                '</div>',
                unsafe_allow_html=True,
            )
            return
        except requests.exceptions.ConnectionError:
            st.markdown(
                f'<div class="alerta-box alerta-box-danger">'
                f'<span class="alerta-titulo alerta-titulo-danger">Sin conexión al servidor</span>'
                f'No se pudo conectar al servicio en <code>{api_url}</code>.'
                f'</div>',
                unsafe_allow_html=True,
            )
            return
        except Exception as e:
            st.markdown(
                f'<div class="alerta-box alerta-box-danger">'
                f'<span class="alerta-titulo alerta-titulo-danger">Error en el análisis</span>'
                f'{str(e)}'
                f'</div>',
                unsafe_allow_html=True,
            )
            return

    # ═══════════════════════════════════════════════════════════════════════
    # SECCIÓN 1: Recomendación principal
    # ═══════════════════════════════════════════════════════════════════════
    st.subheader("Recomendación")

    rec = resultado["recomendacion"]

    if rec["estado"] == "SIN_FORECAST":
        st.markdown(
            f'<div class="alerta-box alerta-box-danger">'
            f'<span class="alerta-titulo alerta-titulo-danger">Sin modelos entrenados</span>'
            f'{rec["mensaje"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    nivel = rec["nivel"]
    if nivel == "ALTA":
        box_class = "alerta-box alerta-box-success"
        titulo_class = "alerta-titulo alerta-titulo-success"
    elif nivel == "MEDIA":
        box_class = "alerta-box"
        titulo_class = "alerta-titulo"
    else:
        box_class = "alerta-box alerta-box-danger"
        titulo_class = "alerta-titulo alerta-titulo-danger"

    st.markdown(
        f'<div class="{box_class}">'
        f'<span class="{titulo_class}">{rec["titulo"]}</span>'
        f'<p style="margin:0.4rem 0 0.8rem 0;">{rec["mensaje"]}</p>'
        f'<p style="margin:0;font-size:0.85rem;color:{colors["muted"]};">'
        f'<strong>Acción sugerida:</strong> {rec["accion_sugerida"]}'
        f'</p>'
        + (
            f'<p style="margin:0.6rem 0 0 0;font-size:0.75rem;color:{colors["muted"]};'
            f'font-style:italic;border-top:1px solid {colors["border"]};'
            f'padding-top:0.5rem;">{rec["nota_confianza"]}</p>'
            if rec.get("nota_confianza") else ""
        )
        + f'</div>',
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # SECCIÓN 2: Gráfica del rango en piezas (INPUT → OUTPUT EN PIEZAS)
    # ═══════════════════════════════════════════════════════════════════════
    st.subheader("Proyección cuantitativa")

    forecast = resultado["forecast"]
    if forecast:
        col_chart, col_kpis = st.columns([2, 1])

        with col_chart:
            fig_forecast = _render_forecast_chart(forecast, rec, colors)
            st.plotly_chart(fig_forecast, use_container_width=True)

        with col_kpis:
            st.metric(
                "Modelo utilizado",
                forecast["modelo_usado"],
                help="Modelo ganador del último análisis de precisión",
            )
            st.metric(
                "Piezas estimadas",
                f"{rec['piezas_estimadas_central']:.0f}",
                help=f"Rango: {rec['piezas_estimadas_rango']['minimo']:.0f}"
                     f" – {rec['piezas_estimadas_rango']['maximo']:.0f}",
            )
            st.metric(
                "Modulador aplicado",
                f"{rec['modulador_aplicado']:.2f}x",
                help="Factor de ajuste según el interés en Google Trends. "
                     ">1 amplifica, <1 reduce.",
            )

    # ═══════════════════════════════════════════════════════════════════════
    # SECCIÓN 3: Gráfica de Google Trends
    # ═══════════════════════════════════════════════════════════════════════
    st.subheader("Interés en Google Trends")

    trends = resultado["analisis_trends"]
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

    with col_kpi1:
        st.metric("Interés promedio", f"{trends['interes_promedio']:.0f}/100")
    with col_kpi2:
        st.metric("Interés máximo", f"{trends['interes_maximo']:.0f}/100")
    with col_kpi3:
        tendencia = trends["pendiente_pct"]
        st.metric("Tendencia reciente", f"{tendencia:+.1f}%",
                  delta="vs meses anteriores")
    with col_kpi4:
        clasif = trends["clasificacion"]
        st.metric("Clasificación", clasif)

    serie = resultado.get("serie_trends", [])
    if serie:
        fig_trends = _render_trends_chart(serie, tema, colors)
        if fig_trends:
            st.plotly_chart(fig_trends, use_container_width=True)
    else:
        st.caption("No hay datos de Trends disponibles para este tema.")

    # ═══════════════════════════════════════════════════════════════════════
    # SECCIÓN 4: Detalles de categoría detectada
    # ═══════════════════════════════════════════════════════════════════════
    st.subheader("Categoría DTF asociada")

    cat = resultado["category_match"]

    col_cat1, col_cat2, col_cat3 = st.columns(3)
    with col_cat1:
        st.metric("Categoría detectada", cat["categoria_detectada"])
    with col_cat2:
        metodo_map = {
            "keyword_match": "Palabra clave",
            "override_manual": "Forzada por usuario",
            "default": "Por defecto",
        }
        st.metric("Método de detección", metodo_map.get(cat["metodo"], cat["metodo"]))
    with col_cat3:
        confianza_map = {"alta": "Alta", "media": "Media", "baja": "Baja"}
        st.metric("Confianza", confianza_map.get(cat["confianza"], cat["confianza"]))

    if cat["confianza"] == "baja":
        st.markdown(
            f'<div class="alerta-box" style="margin-top:0.8rem;">'
            f'<span class="alerta-titulo">Confianza baja en detección</span>'
            f'No se detectaron palabras clave específicas en el tema. El sistema usó '
            f'"Casual" como categoría por defecto. Si tienes claro el tipo de producto '
            f'que planeas, usa el selector arriba para forzar la categoría correcta.'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # SECCIÓN 5: Metodología (transparencia académica)
    # ═══════════════════════════════════════════════════════════════════════
    with st.expander("¿Cómo se calculó esta recomendación?"):
        st.markdown(
            f'<div class="alerta-box">'
            f'<span class="alerta-titulo">Metodología</span>'
            f'<p style="margin:0.4rem 0;">'
            f'1. <strong>Google Trends</strong> mide el interés público en el tema '
            f'(últimos 12 meses, geo={resultado["metadata"]["geo"]}).'
            f'</p>'
            f'<p style="margin:0.4rem 0;">'
            f'2. <strong>Modelo {forecast["modelo_usado"] if forecast else "ML"}</strong> '
            f'proyecta las ventas base para los próximos {horizonte_dias} días '
            f'usando el historial completo de la marca + transfer learning H&M.'
            f'</p>'
            f'<p style="margin:0.4rem 0;">'
            f'3. <strong>Modulador</strong> ajusta el pronóstico base según el interés '
            f'del tema: temas calientes amplifican (1.3x), temas fríos reducen (0.6x).'
            f'</p>'
            f'<p style="margin:0.4rem 0;font-size:0.8rem;color:{colors["muted"]};'
            f'font-style:italic;border-top:1px solid {colors["border"]};padding-top:0.5rem;">'
            f'A diferencia del Analizador de Diseños (heurística sobre imagen), '
            f'esta recomendación usa el modelo ML entrenado, cuya precisión '
            f'está validada empíricamente (MAPE 15.48%).'
            f'</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Datos técnicos de la evaluación"):
        st.json(resultado.get("metadata", {}))