"""
Dashboard v5.0 — DTF Fashion
Diseño fashion, lenguaje para administrador de marca.
Tabs: Demanda Estimada · Precisión del Análisis · Plan de Producción
      Tendencias de Mercado · Historial de Ventas · Avisos del Sistema
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.connection import read_sql, engine

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dashboard")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DTF Fashion — Plataforma de Demanda",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ─────────────────────────────────────────────────────────────────────────────
# TEMA — estado persistente entre reruns
# ─────────────────────────────────────────────────────────────────────────────

if "tema" not in st.session_state:
    st.session_state.tema = "dark"

ES_OSCURO = st.session_state.tema == "dark"

# ─────────────────────────────────────────────────────────────────────────────
# PALETA Y ESTILOS — dinámicos según el tema activo
# ─────────────────────────────────────────────────────────────────────────────

if ES_OSCURO:
    COLORS = {
        "gold":        "#C2A87A",
        "gold_dim":    "rgba(194,168,122,0.15)",
        "teal":        "#4E9E8A",
        "teal_dim":    "rgba(78,158,138,0.15)",
        "amber":       "#C4936A",
        "red":         "#B05C5C",
        "surface":     "#16161E",
        "border":      "#2A2A38",
        "text":        "#E8E4DC",
        "muted":       "#8A8898",
        "banda":       "rgba(194,168,122,0.12)",
        # internos CSS
        "_bg":         "#0C0C10",
        "_sidebar":    "#0F0F14",
        "_plot":       "rgba(22,22,30,0.6)",
        "_h2":         "#C2A87A",
        "_h3":         "#E8E4DC",
        "_tab_active": "#C2A87A",
        "_template":   "plotly_dark",
    }
else:
    COLORS = {
        "gold":        "#8B6310",
        "gold_dim":    "rgba(139,99,16,0.12)",
        "teal":        "#2D7A68",
        "teal_dim":    "rgba(45,122,104,0.12)",
        "amber":       "#A0622A",
        "red":         "#8B3A3A",
        "surface":     "#EEE9E2",
        "border":      "#D4CFC7",
        "text":        "#1C1C28",
        "muted":       "#7A7060",
        "banda":       "rgba(139,99,16,0.10)",
        # internos CSS
        "_bg":         "#FAF8F4",
        "_sidebar":    "#EAE5DD",
        "_plot":       "rgba(240,236,229,0.6)",
        "_h2":         "#8B6310",
        "_h3":         "#1C1C28",
        "_tab_active": "#8B6310",
        "_template":   "plotly_white",
    }

MODELOS_VISIBLES = ["SARIMA", "Prophet", "Random Forest"]

CHART_LAYOUT = dict(
    template=COLORS["_template"],
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=COLORS["_plot"],
    font=dict(family="sans-serif", color=COLORS["text"], size=12),
    xaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"]),
    yaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"]),
)

MARGIN_DEFAULT = dict(l=20, r=20, t=50, b=20)
MARGIN_LEGEND_BOTTOM = dict(l=20, r=20, t=50, b=80)

LEGEND_DEFAULT = dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["border"])
LEGEND_TOP = dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["border"],
                  orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)

_C = COLORS  # alias corto para el bloque CSS
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* Ocultar elementos de Streamlit */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
[data-testid="stDecoration"] {{display: none;}}

/* Fondo global */
.stApp {{
    background-color: {_C["_bg"]} !important;
}}

/* ── COLOR DE TEXTO GLOBAL ───────────────────────────────────────────────────
   Pisa el textColor de config.toml (que siempre es el valor dark #E8E4DC).
   Se aplica a todos los elementos de texto nativos de Streamlit. */
body,
p,
li,
span:not(.badge-ok):not(.badge-warn):not(.badge-danger):not(.alerta-titulo):not(.ts-label),
.stMarkdown,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stText"],
[data-testid="stWidgetLabel"],
[data-testid="stCaptionContainer"],
[data-testid="stSelectbox"] *,
[data-testid="stMultiSelect"] *,
[data-testid="stSlider"] *,
[data-testid="stCheckbox"] *,
[data-testid="stRadio"] *,
[data-testid="stNumberInput"] *,
[data-testid="stTextInput"] *,
[data-testid="stFileUploader"] *,
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th,
[data-baseweb="tab"] span,
table td, table th {{
    color: {_C["text"]} !important;
}}
/* Sidebar texto nativo */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not(.badge-ok):not(.badge-warn):not(.badge-danger),
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
    color: {_C["text"]} !important;
}}

/* Contenedor principal */
.main .block-container {{
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1440px;
    background: transparent;
}}

/* Métricas */
[data-testid="stMetric"] {{
    background: {_C["surface"]};
    border: 1px solid {_C["border"]};
    border-radius: 2px;
    padding: 1.2rem 1.4rem;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {_C["muted"]} !important;
    font-weight: 500;
}}
[data-testid="stMetricValue"] {{
    font-size: 1.8rem;
    font-weight: 300;
    color: {_C["text"]} !important;
}}
[data-testid="stMetricDelta"] {{
    font-size: 0.78rem;
}}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    gap: 0;
    border-bottom: 1px solid {_C["border"]};
    background: transparent;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 500;
    color: {_C["muted"]};
    padding: 0.8rem 1.4rem;
    border-bottom: 2px solid transparent;
    background: transparent;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    color: {_C["_tab_active"]} !important;
    border-bottom: 2px solid {_C["_tab_active"]} !important;
    background: transparent !important;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {_C["_sidebar"]};
    border-right: 1px solid {_C["border"]};
}}
[data-testid="stSidebar"] .block-container {{
    padding-top: 2rem;
}}

/* Botones */
.stButton > button {{
    border-radius: 1px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 0.75rem;
    border: 1px solid {_C["border"]};
    color: {_C["text"]};
    background: transparent;
    transition: all 0.2s ease;
}}
.stButton > button[kind="primary"] {{
    background: {_C["gold"]};
    border-color: {_C["gold"]};
    color: {_C["_bg"]};
}}
.stButton > button[kind="primary"]:hover {{
    opacity: 0.85;
}}
.stButton > button:hover {{
    border-color: {_C["gold"]};
    color: {_C["gold"]};
}}

/* Dividers */
hr {{
    border-color: {_C["border"]} !important;
    margin: 1.5rem 0 !important;
}}

/* Cabeceras */
h1 {{
    font-weight: 300 !important;
    letter-spacing: 0.05em !important;
    font-size: 1.6rem !important;
    color: {_C["text"]} !important;
}}
h2 {{
    font-weight: 400 !important;
    letter-spacing: 0.04em !important;
    font-size: 1.1rem !important;
    color: {_C["_h2"]} !important;
    text-transform: uppercase;
    margin-top: 2rem !important;
}}
h3 {{
    font-weight: 400 !important;
    font-size: 0.95rem !important;
    color: {_C["_h3"]} !important;
}}

/* Captions / timestamp */
.ts-label {{
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    color: {_C["muted"]};
    text-transform: uppercase;
    margin-bottom: 1.5rem;
    display: block;
}}

/* Badges */
.badge-ok {{
    display: inline-block;
    background: {_C["teal_dim"]};
    border: 1px solid {_C["teal"]};
    color: {_C["teal"]};
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.7rem;
    border-radius: 1px;
    font-weight: 500;
}}
.badge-warn {{
    display: inline-block;
    background: {_C["gold_dim"]};
    border: 1px solid {_C["amber"]};
    color: {_C["amber"]};
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.7rem;
    border-radius: 1px;
    font-weight: 500;
}}
.badge-danger {{
    display: inline-block;
    background: rgba(176,92,92,0.12);
    border: 1px solid {_C["red"]};
    color: {_C["red"]};
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.7rem;
    border-radius: 1px;
    font-weight: 500;
}}

/* Alertas personalizadas */
.alerta-box {{
    background: {_C["surface"]};
    border: 1px solid {_C["border"]};
    border-left: 3px solid {_C["gold"]};
    border-radius: 2px;
    padding: 1rem 1.4rem;
    margin-bottom: 1rem;
    font-size: 0.88rem;
    color: {_C["text"]};
    line-height: 1.6;
}}
.alerta-box-danger {{
    border-left-color: {_C["red"]};
}}
.alerta-box-success {{
    border-left-color: {_C["teal"]};
}}
.alerta-titulo {{
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {_C["gold"]};
    font-weight: 600;
    margin-bottom: 0.4rem;
    display: block;
}}
.alerta-titulo-danger {{
    color: {_C["red"]};
}}
.alerta-titulo-success {{
    color: {_C["teal"]};
}}

/* Tabla de datos */
[data-testid="stDataFrame"] {{
    border: 1px solid {_C["border"]};
    border-radius: 2px;
}}

/* Selectbox */
[data-testid="stSelectbox"] label {{
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {_C["muted"]};
}}

/* File uploader */
[data-testid="stFileUploader"] label {{
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {_C["muted"]};
}}

/* Expander */
[data-testid="stExpander"] {{
    border: 1px solid {_C["border"]} !important;
    border-radius: 2px !important;
    background: {_C["surface"]} !important;
}}

/* Selectbox / input backgrounds en modo claro */
[data-baseweb="select"] div,
[data-baseweb="input"] input {{
    background: {_C["surface"]} !important;
    color: {_C["text"]} !important;
    border-color: {_C["border"]} !important;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def cargar_serie():
    try:
        return read_sql("SELECT * FROM serie_semanal ORDER BY fecha")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def cargar_predicciones():
    try:
        df = read_sql("""
            SELECT p.*, t.modelo_ganador
            FROM predicciones p
            LEFT JOIN training_runs t ON p.run_id = t.run_id
            ORDER BY p.modelo, p.fecha_prediccion
        """)
        if not df.empty:
            df = df[df["modelo"].isin(MODELOS_VISIBLES)]
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def cargar_metricas():
    try:
        runs = read_sql("SELECT * FROM training_runs ORDER BY fecha_ejecucion DESC LIMIT 1")
        if runs.empty:
            return pd.DataFrame(), {}
        run = runs.iloc[0]
        metricas = read_sql(
            f"SELECT * FROM metricas_modelos WHERE run_id = '{run['run_id']}' ORDER BY mape"
        )
        if not metricas.empty:
            metricas = metricas[metricas["modelo"].isin(MODELOS_VISIBLES)]
        return metricas, run.to_dict()
    except Exception:
        return pd.DataFrame(), {}


@st.cache_data(ttl=60)
def cargar_ventas():
    try:
        return read_sql("SELECT * FROM ventas ORDER BY fecha")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def cargar_ultimo_entrenamiento():
    try:
        runs = read_sql("SELECT fecha_ejecucion FROM training_runs ORDER BY fecha_ejecucion DESC LIMIT 1")
        if runs.empty:
            return None
        return pd.to_datetime(runs.iloc[0]["fecha_ejecucion"])
    except Exception:
        return None


def check_db():
    try:
        read_sql("SELECT 1 AS ok")
        return True
    except Exception:
        return False


def timestamp_label(prefix="Datos actualizados"):
    ahora = datetime.now().strftime("%d %b %Y, %H:%M")
    return f'<span class="ts-label">{prefix} · {ahora}</span>'


def ganador_visible(pred, metricas_df, run_info):
    """Devuelve el modelo ganador entre los visibles."""
    g = run_info.get("modelo_ganador", "")
    if g in MODELOS_VISIBLES:
        return g
    if not metricas_df.empty:
        return metricas_df.iloc[0]["modelo"]
    if not pred.empty:
        return pred[pred["modelo"].isin(MODELOS_VISIBLES)].groupby(
            "modelo"
        )["unidades_predichas"].mean().idxmin()
    return "SARIMA"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### DTF Fashion")
    st.markdown(
        f'<span style="font-size:0.75rem;letter-spacing:0.08em;'
        f'text-transform:uppercase;color:{COLORS["muted"]};">Plataforma de demanda</span>',
        unsafe_allow_html=True,
    )
    st.divider()

    db_ok = check_db()
    if db_ok:
        st.markdown('<span class="badge-ok">Base de datos conectada</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-danger">Sin conexion a base de datos</span>', unsafe_allow_html=True)
        st.caption("Verifica que PostgreSQL este corriendo.")

    # Alerta de modelo obsoleto en sidebar
    ultimo = cargar_ultimo_entrenamiento()
    if ultimo is not None:
        dias_sin_entrenar = (datetime.now() - ultimo).days
        if dias_sin_entrenar > 30:
            st.markdown(
                f'<div style="margin-top:0.8rem;">'
                f'<span class="badge-warn">Modelo con {dias_sin_entrenar} dias sin actualizar</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    st.markdown(
        f'<span style="font-size:0.72rem;letter-spacing:0.1em;'
        f'text-transform:uppercase;color:{COLORS["muted"]};">Cargar ventas</span>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Archivo Excel o CSV",
        type=["xlsx", "xls", "csv"],
        label_visibility="collapsed",
        help="Columnas requeridas: fecha, cantidad. Opcionales: precio, producto, categoria.",
    )

    if uploaded_file:
        if st.button("Procesar archivo", type="primary", use_container_width=True):
            with st.spinner("Procesando datos..."):
                try:
                    import tempfile
                    tmp = Path(tempfile.mkdtemp()) / uploaded_file.name
                    tmp.write_bytes(uploaded_file.read())
                    from etl.etl_pipeline import ejecutar_pipeline
                    resultado = ejecutar_pipeline(str(tmp))
                    st.success(f"{resultado['filas_limpias']} registros cargados correctamente.")
                    st.cache_data.clear()
                    import shutil
                    shutil.rmtree(tmp.parent, ignore_errors=True)
                except Exception as e:
                    st.error(f"Error al procesar: {e}")

    st.divider()

    st.markdown(
        f'<span style="font-size:0.72rem;letter-spacing:0.1em;'
        f'text-transform:uppercase;color:{COLORS["muted"]};">Actualizar pronósticos</span>',
        unsafe_allow_html=True,
    )
    if st.button("Ejecutar análisis de modelos", use_container_width=True):
        with st.spinner("Ejecutando modelos... puede tardar 1-2 minutos."):
            try:
                from models.train_models import ejecutar_entrenamiento
                resultado = ejecutar_entrenamiento()
                if resultado["status"] == "ok":
                    st.success(f"Modelo mas preciso: {resultado['ganador']}")
                    st.cache_data.clear()
                else:
                    st.error(resultado.get("mensaje", "Error desconocido."))
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # ── Toggle de tema ────────────────────────────────────────────────────────
    icono_tema  = "Fondo claro" if ES_OSCURO else "Fondo oscuro"
    if st.button(icono_tema, use_container_width=True, key="btn_tema"):
        st.session_state.tema = "light" if ES_OSCURO else "dark"
        st.rerun()

    st.divider()
    st.markdown(
        f'<span style="font-size:0.68rem;color:{COLORS["muted"]};">'
        'v5.0 — Proyecto de titulacion Ibero 2026<br>'
        'Modelos: SARIMA · Prophet · Random Forest'
        '</span>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Demanda Estimada",
    "Precisión del Análisis",
    "Plan de Producción",
    "Tendencias de Mercado",
    "Historial de Ventas",
    "Avisos del Sistema",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — DEMANDA ESTIMADA
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Demanda Estimada — Próximos 30 días")
    st.markdown(timestamp_label(), unsafe_allow_html=True)

    pred = cargar_predicciones()
    serie = cargar_serie()
    metricas_df, run_info = cargar_metricas()

    if pred.empty or serie.empty:
        st.markdown(
            '<div class="alerta-box">'
            '<span class="alerta-titulo">Sin datos de pronostico</span>'
            'Carga tu archivo de ventas desde el panel lateral y ejecuta el analisis de modelos '
            'para ver la demanda estimada.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        pred["fecha_prediccion"] = pd.to_datetime(pred["fecha_prediccion"])
        serie["fecha"] = pd.to_datetime(serie["fecha"])

        ganador = ganador_visible(pred, metricas_df, run_info)
        pg = pred[pred["modelo"] == ganador]

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Piezas estimadas (30 días)", f"{pg['unidades_predichas'].sum():.0f}")
        with col2:
            ri = pg["banda_inferior"].sum()
            rs = pg["banda_superior"].sum()
            st.metric("Rango esperado", f"{ri:.0f} – {rs:.0f}")
        with col3:
            st.metric("Promedio por día", f"{pg['unidades_predichas'].mean():.1f} pzas")
        with col4:
            st.metric("Modelo activo", ganador)

        st.divider()

        # Grafica principal: historico + forecast con banda sombreada
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=serie["fecha"],
            y=serie["unidades"],
            name="Ventas reales",
            marker_color=COLORS["gold"],
            opacity=0.45,
        ))

        # Banda sombreada ±30%
        fig.add_trace(go.Scatter(
            x=pd.concat([
                pg["fecha_prediccion"],
                pg["fecha_prediccion"].iloc[::-1],
            ]),
            y=pd.concat([
                pg["banda_superior"],
                pg["banda_inferior"].iloc[::-1],
            ]),
            fill="toself",
            fillcolor=COLORS["banda"],
            line=dict(color="rgba(0,0,0,0)"),
            name="Rango estimado (±30%)",
            showlegend=True,
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=pg["fecha_prediccion"],
            y=pg["unidades_predichas"],
            name=f"Pronostico — {ganador}",
            mode="lines+markers",
            line=dict(color=COLORS["gold"], width=2),
            marker=dict(size=4, color=COLORS["gold"]),
        ))

        fig.update_layout(
            **CHART_LAYOUT,
            title="Ventas históricas y pronóstico",
            xaxis_title=None,
            yaxis_title="Piezas",
            height=480,
            hovermode="x unified",
            margin=MARGIN_LEGEND_BOTTOM,
            legend=dict(
                orientation="h",
                yanchor="top", y=-0.12,
                xanchor="center", x=0.5,
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Selector de modelo
        st.subheader("Comparar por modelo")
        modelos_disp = [m for m in pred["modelo"].unique() if m in MODELOS_VISIBLES]
        modelo_sel = st.selectbox("Modelo:", modelos_disp, index=0, label_visibility="collapsed")

        pred_sel = pred[pred["modelo"] == modelo_sel]
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(f"Total estimado — {modelo_sel}", f"{pred_sel['unidades_predichas'].sum():.0f} pzas")
        with col_b:
            st.metric("Promedio diario", f"{pred_sel['unidades_predichas'].mean():.1f} pzas")

        with st.expander("Ver detalle día por día"):
            tabla = pred_sel[[
                "fecha_prediccion", "unidades_predichas",
                "banda_inferior", "banda_superior", "dia_horizonte",
            ]].copy()
            tabla.columns = ["Fecha", "Piezas estimadas", "Mínimo", "Máximo", "Día"]
            tabla["Fecha"] = tabla["Fecha"].dt.strftime("%d %b %Y")
            st.dataframe(tabla.reset_index(drop=True), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRECISION DEL ANALISIS
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Precisión del Análisis")
    st.markdown(timestamp_label("Ultimo entrenamiento"), unsafe_allow_html=True)

    metricas_df, run_info = cargar_metricas()

    if metricas_df.empty:
        st.markdown(
            '<div class="alerta-box">'
            '<span class="alerta-titulo">Sin analisis ejecutado</span>'
            'Ejecuta el analisis de modelos desde el panel lateral para ver los resultados de precision.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        pred = cargar_predicciones()
        ganador = ganador_visible(pred, metricas_df, run_info)

        # KPIs principales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Modelo más preciso", ganador)
        with col2:
            baseline_mape = run_info.get("baseline_mape", 0)
            st.metric(
                "Error sin modelo de IA",
                f"{baseline_mape:.1f}%",
                help="Que tan impreciso seria pronosticar sin inteligencia artificial",
            )
        with col3:
            mejora = run_info.get("mejora_pct", 0)
            delta_txt = "Objetivo superado (>20%)" if mejora and mejora >= 20 else "Por debajo del objetivo"
            st.metric(
                "Mejora con IA",
                f"{mejora:+.1f}%" if mejora else "N/D",
                delta=delta_txt,
            )

        st.divider()

        # Grafica comparativa de modelos
        es_ganador_mask = metricas_df["modelo"] == ganador
        bar_colors = [COLORS["gold"] if g else COLORS["muted"] for g in es_ganador_mask]

        fig_comp = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Margen de error (menor = mejor)", "Diferencia promedio en piezas (menor = mejor)"],
        )

        fig_comp.add_trace(go.Bar(
            x=metricas_df["modelo"],
            y=metricas_df["mape"],
            marker_color=bar_colors,
            name="Margen de error",
            showlegend=False,
            text=(metricas_df["mape"].round(1).astype(str) + "%"),
            textposition="outside",
            textfont=dict(color=COLORS["text"], size=11),
        ), row=1, col=1)

        fig_comp.add_trace(go.Bar(
            x=metricas_df["modelo"],
            y=metricas_df["mae"],
            marker_color=bar_colors,
            name="Diferencia promedio",
            showlegend=False,
            text=metricas_df["mae"].round(2),
            textposition="outside",
            textfont=dict(color=COLORS["text"], size=11),
        ), row=1, col=2)

        if baseline_mape:
            fig_comp.add_hline(
                y=baseline_mape,
                line_dash="dash",
                line_color=COLORS["red"],
                annotation_text=f"Sin IA: {baseline_mape:.1f}%",
                annotation_font_color=COLORS["red"],
                row=1, col=1,
            )

        fig_comp.update_layout(
            **CHART_LAYOUT,
            margin=MARGIN_DEFAULT,
            height=380,
            title="Desempeño de los modelos de pronóstico",
        )
        fig_comp.update_xaxes(gridcolor=COLORS["border"])
        fig_comp.update_yaxes(gridcolor=COLORS["border"])
        st.plotly_chart(fig_comp, use_container_width=True)

        # Tabla simplificada
        st.subheader("Resumen de modelos")
        tabla_met = metricas_df[["modelo", "mape", "mae", "es_ganador"]].copy()
        tabla_met["mape"] = tabla_met["mape"].apply(lambda x: f"{x:.1f}%")
        tabla_met["mae"] = tabla_met["mae"].apply(lambda x: f"{x:.2f} pzas")
        tabla_met["es_ganador"] = tabla_met["es_ganador"].apply(lambda x: "Activo" if x else "")
        tabla_met.columns = ["Modelo", "Margen de error", "Diferencia promedio", "Estado"]
        st.dataframe(tabla_met.reset_index(drop=True), use_container_width=True, hide_index=True)

        # Forecast comparativo
        st.subheader("Pronóstico según cada modelo")
        pred = cargar_predicciones()
        if not pred.empty:
            pred["fecha_prediccion"] = pd.to_datetime(pred["fecha_prediccion"])
            fig_all = go.Figure()
            line_styles = ["solid", "dash", "dot"]
            palette = [COLORS["gold"], COLORS["teal"], COLORS["amber"]]

            for i, (modelo, grupo) in enumerate(pred.groupby("modelo")):
                if modelo not in MODELOS_VISIBLES:
                    continue
                es_winner = modelo == ganador
                fig_all.add_trace(go.Scatter(
                    x=grupo["fecha_prediccion"],
                    y=grupo["unidades_predichas"],
                    name=modelo + (" (activo)" if es_winner else ""),
                    mode="lines+markers",
                    line=dict(
                        width=2.5 if es_winner else 1.5,
                        dash=line_styles[i % len(line_styles)],
                        color=palette[i % len(palette)],
                    ),
                    marker=dict(size=4 if es_winner else 2, color=palette[i % len(palette)]),
                ))

            fig_all.update_layout(
                **CHART_LAYOUT,
                margin=MARGIN_DEFAULT,
                title="Los tres modelos sobre el mismo horizonte",
                yaxis_title="Piezas estimadas",
                height=380,
                hovermode="x unified",
            )
            st.plotly_chart(fig_all, use_container_width=True)

        with st.expander("Historial de análisis ejecutados"):
            try:
                runs = read_sql("SELECT * FROM training_runs ORDER BY fecha_ejecucion DESC LIMIT 20")
                if not runs.empty:
                    runs_display = runs[[
                        "run_id", "fecha_ejecucion", "modelo_ganador",
                        "n_datos", "mejor_mape", "mejora_pct",
                    ]].copy()
                    runs_display.columns = [
                        "ID", "Fecha", "Modelo activo", "Días de datos", "Error (%)", "Mejora vs sin IA (%)"
                    ]
                    st.dataframe(runs_display, use_container_width=True, hide_index=True)
            except Exception:
                st.caption("Sin historial disponible.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — PLAN DE PRODUCCION
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Plan de Producción")
    st.markdown(timestamp_label(), unsafe_allow_html=True)

    pred = cargar_predicciones()
    serie = cargar_serie()
    ventas = cargar_ventas()
    metricas_df, run_info = cargar_metricas()

    if pred.empty:
        st.markdown(
            '<div class="alerta-box">'
            '<span class="alerta-titulo">Sin plan de produccion</span>'
            'Ejecuta el analisis de modelos para generar el plan de produccion sugerido.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        pred["fecha_prediccion"] = pd.to_datetime(pred["fecha_prediccion"])
        serie["fecha"] = pd.to_datetime(serie["fecha"])

        ganador = ganador_visible(pred, metricas_df, run_info)
        pg = pred[pred["modelo"] == ganador].copy()

        # Escenarios
        st.subheader("Proyección de piezas — próximos 30 días")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Escenario conservador",
                f"{pg['banda_inferior'].sum():.0f} pzas",
                help="Producción mínima recomendada para cubrir demanda baja",
            )
        with col2:
            st.metric(
                "Escenario central",
                f"{pg['unidades_predichas'].sum():.0f} pzas",
                help="Estimación más probable según el modelo",
            )
        with col3:
            st.metric(
                "Escenario optimista",
                f"{pg['banda_superior'].sum():.0f} pzas",
                help="Producción máxima para aprovechar demanda alta",
            )

        st.divider()

        # Desglose semanal
        pg["semana"] = pg["fecha_prediccion"].dt.isocalendar().week.astype(int)
        pg["semana_label"] = pg["fecha_prediccion"].dt.strftime("Sem %d %b")
        semanal = (
            pg.groupby(pg["fecha_prediccion"].dt.to_period("W").dt.start_time)
            .agg(
                unidades=("unidades_predichas", "sum"),
                inferior=("banda_inferior", "sum"),
                superior=("banda_superior", "sum"),
            )
            .reset_index()
        )
        semanal.columns = ["semana_inicio", "unidades", "inferior", "superior"]
        semanal["label"] = semanal["semana_inicio"].dt.strftime("Sem %d %b")

        fig_sem = go.Figure()
        fig_sem.add_trace(go.Bar(
            x=semanal["label"],
            y=semanal["unidades"],
            name="Estimacion central",
            marker_color=COLORS["gold"],
            opacity=0.85,
            text=semanal["unidades"].round(0).astype(int),
            textposition="outside",
            textfont=dict(color=COLORS["text"], size=11),
        ))
        fig_sem.add_trace(go.Scatter(
            x=semanal["label"],
            y=semanal["superior"],
            mode="markers",
            marker=dict(color=COLORS["teal"], size=9, symbol="triangle-up"),
            name="Optimista (+30%)",
        ))
        fig_sem.add_trace(go.Scatter(
            x=semanal["label"],
            y=semanal["inferior"],
            mode="markers",
            marker=dict(color=COLORS["amber"], size=9, symbol="triangle-down"),
            name="Conservador (-30%)",
        ))
        fig_sem.update_layout(
            **CHART_LAYOUT,
            margin=MARGIN_DEFAULT,
            title="Producción sugerida por semana",
            yaxis_title="Piezas",
            height=400,
        )
        st.plotly_chart(fig_sem, use_container_width=True)

        # Analisis por categoria
        if not ventas.empty and "categoria" in ventas.columns:
            st.subheader("Rendimiento por categoría")
            col_a, col_b = st.columns(2)

            cat_vol = ventas.groupby("categoria")["cantidad"].sum().sort_values(ascending=False)
            cat_ing = ventas.groupby("categoria")["ingreso_bruto"].sum().sort_values(ascending=False)

            with col_a:
                fig_cat = go.Figure(go.Pie(
                    values=cat_vol.values,
                    labels=cat_vol.index,
                    hole=0.5,
                    marker=dict(
                        colors=px.colors.sequential.Oranges[2:],
                        line=dict(color=COLORS["border"], width=1),
                    ),
                    textfont=dict(size=11),
                ))
                fig_cat.update_layout(
                    **CHART_LAYOUT,
                    margin=MARGIN_DEFAULT,
                    title="Volumen por categoría",
                    height=360,
                    showlegend=True,
                )
                st.plotly_chart(fig_cat, use_container_width=True)

            with col_b:
                fig_ing = go.Figure(go.Bar(
                    x=cat_ing.index,
                    y=cat_ing.values,
                    marker=dict(
                        color=cat_ing.values,
                        colorscale=[[0, COLORS["border"]], [1, COLORS["gold"]]],
                    ),
                    text=cat_ing.apply(lambda x: f"${x:,.0f}"),
                    textposition="outside",
                    textfont=dict(size=10, color=COLORS["text"]),
                ))
                fig_ing.update_layout(
                    **CHART_LAYOUT,
                    margin=MARGIN_DEFAULT,
                    title="Ingreso por categoria (MXN)",
                    yaxis_title="Ingresos",
                    height=360,
                    showlegend=False,
                )
                st.plotly_chart(fig_ing, use_container_width=True)

        # Recomendaciones colapsables
        prom_hist = serie["unidades"].mean() if not serie.empty else 0
        prom_pred = pg["unidades_predichas"].mean()
        cambio = ((prom_pred - prom_hist) / prom_hist * 100) if prom_hist > 0 else 0

        pg_copy = pg.copy()
        pg_copy["dia_semana"] = pg_copy["fecha_prediccion"].dt.dayofweek
        mejor_dia_num = pg_copy.groupby("dia_semana")["unidades_predichas"].mean().idxmax()
        dias_nombre = {0: "lunes", 1: "martes", 2: "miércoles", 3: "jueves",
                       4: "viernes", 5: "sábado", 6: "domingo"}
        cat_top = ""
        if not ventas.empty and "categoria" in ventas.columns:
            cat_top = ventas.groupby("categoria")["cantidad"].sum().idxmax()

        with st.expander("Ver recomendaciones del período"):
            if cambio > 20:
                st.markdown(
                    f'<div class="alerta-box alerta-box-success">'
                    f'<span class="alerta-titulo alerta-titulo-success">Oportunidad de demanda</span>'
                    f'La demanda estimada supera tu promedio histórico en <strong>{cambio:.0f}%</strong>. '
                    f'Considera asegurar insumos DTF adicionales para no perder ventas en el período.'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            elif cambio < -10:
                st.markdown(
                    f'<div class="alerta-box alerta-box-danger">'
                    f'<span class="alerta-titulo alerta-titulo-danger">Demanda por debajo del promedio</span>'
                    f'La demanda estimada está <strong>{abs(cambio):.0f}%</strong> por debajo de tu histórico. '
                    f'Reduce el volumen de producción para evitar inventario sin venta.'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="alerta-box alerta-box-success">'
                    f'<span class="alerta-titulo alerta-titulo-success">Demanda estable</span>'
                    f'La demanda se mantiene en niveles similares a tu histórico (<strong>{cambio:+.0f}%</strong>). '
                    f'Mantén tu plan de producción habitual.'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="alerta-box">'
                f'<span class="alerta-titulo">Mejor día para lanzamientos</span>'
                f'El <strong>{dias_nombre.get(mejor_dia_num, "?")}</strong> concentra la mayor demanda semanal según el pronóstico. '
                f'Programa lanzamientos y promociones ese día para maximizar ventas.'
                f'</div>',
                unsafe_allow_html=True,
            )

            if cat_top:
                cat_ing = ventas.groupby("categoria")["ingreso_bruto"].sum().idxmax()
                st.markdown(
                    f'<div class="alerta-box">'
                    f'<span class="alerta-titulo">Categorías clave</span>'
                    f'La categoría <strong>{cat_top}</strong> lidera en volumen de piezas vendidas '
                    f'y <strong>{cat_ing}</strong> lidera en ingresos. '
                    f'Prioriza stock e insumos DTF para estas categorías.'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — TENDENCIAS DE MERCADO
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Tendencias de Mercado")
    st.markdown(timestamp_label(), unsafe_allow_html=True)

    st.caption(
        "Detecta tendencias emergentes de diseno consultando Google Trends en tiempo real."
    )

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        keywords_input = st.text_input(
            "Terminos de busqueda (maximo 5, separados por coma)",
            value="playera personalizada, DTF printing, diseno streetwear",
            label_visibility="visible",
        )
    with col2:
        timeframe = st.selectbox(
            "Periodo",
            ["today 1-m", "today 3-m", "today 12-m"],
            index=1,
            format_func=lambda x: {"today 1-m": "1 mes", "today 3-m": "3 meses", "today 12-m": "12 meses"}[x],
        )
    with col3:
        geo = st.selectbox(
            "Pais",
            ["MX", "US", "CO", "AR", "ES", ""],
            index=0,
            format_func=lambda x: x if x else "Global",
        )

    if st.button("Buscar tendencias", type="primary"):
        with st.spinner("Consultando Google Trends..."):
            try:
                from pytrends.request import TrendReq
                kw_list = [k.strip() for k in keywords_input.split(",")][:5]
                pytrends = TrendReq(hl="es-MX", tz=360, timeout=(10, 25))
                pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo=geo)
                interest = pytrends.interest_over_time()

                if interest.empty:
                    st.markdown(
                        '<div class="alerta-box">'
                        '<span class="alerta-titulo">Sin resultados</span>'
                        'Google no devolvio datos para estos terminos en el periodo seleccionado. '
                        'Intenta con terminos diferentes o un periodo mas amplio.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    if "isPartial" in interest.columns:
                        interest = interest.drop("isPartial", axis=1)

                    palette_trends = [COLORS["gold"], COLORS["teal"], COLORS["amber"],
                                      "#8A7FBE", "#B05C5C"]
                    fig_trends = go.Figure()
                    for i, kw in enumerate(kw_list):
                        if kw in interest.columns:
                            fig_trends.add_trace(go.Scatter(
                                x=interest.index,
                                y=interest[kw],
                                name=kw,
                                mode="lines",
                                line=dict(color=palette_trends[i % len(palette_trends)], width=2),
                            ))
                    fig_trends.update_layout(
                        **CHART_LAYOUT,
                        margin=MARGIN_DEFAULT,
                        title="Interés en el tiempo — Google Trends",
                        yaxis_title="Interés relativo (0 a 100)",
                        height=420,
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_trends, use_container_width=True)

                    st.subheader("Resumen del período")
                    summary = interest.describe().loc[["mean", "max", "min"]].round(1)
                    summary.index = ["Promedio", "Máximo", "Mínimo"]
                    st.dataframe(summary, use_container_width=True)

                    try:
                        related = pytrends.related_queries()
                        for kw in kw_list:
                            if kw in related and related[kw]["top"] is not None:
                                with st.expander(f"Busquedas relacionadas con: {kw}"):
                                    st.dataframe(
                                        related[kw]["top"].head(10),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                    except Exception:
                        pass

            except ImportError:
                st.error("pytrends no esta instalado.")
            except Exception as e:
                st.error(f"Error al consultar Google Trends: {e}")



# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — HISTORIAL DE VENTAS
# ═════════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("Historial de Ventas")
    st.markdown(timestamp_label(), unsafe_allow_html=True)

    serie = cargar_serie()
    ventas = cargar_ventas()

    if serie.empty:
        st.markdown(
            '<div class="alerta-box">'
            '<span class="alerta-titulo">Sin datos cargados</span>'
            'Carga tu archivo Excel o CSV de ventas desde el panel lateral para visualizar tu historial.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        serie["fecha"] = pd.to_datetime(serie["fecha"])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total piezas vendidas", f"{serie['unidades'].sum():.0f}")
        with col2:
            st.metric("Ingreso total", f"${serie['ingreso_bruto'].sum():,.0f} MXN")
        with col3:
            dias_activos = (serie["unidades"] > 0).sum()
            st.metric("Dias con ventas", f"{dias_activos} de {len(serie)}")
        with col4:
            if not ventas.empty and "producto" in ventas.columns:
                st.metric("Productos distintos", ventas["producto"].nunique())

        st.divider()

        # Grafica historico
        fig_hist = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Piezas vendidas por día", "Ingresos acumulados (MXN)"],
            shared_xaxes=True,
            vertical_spacing=0.14,
            row_heights=[0.6, 0.4],
        )

        fig_hist.add_trace(go.Bar(
            x=serie["fecha"],
            y=serie["unidades"],
            name="Piezas",
            marker_color=COLORS["gold"],
            opacity=0.8,
        ), row=1, col=1)

        if "ingreso_acumulado" in serie.columns:
            fig_hist.add_trace(go.Scatter(
                x=serie["fecha"],
                y=serie["ingreso_acumulado"],
                name="Ingreso acumulado",
                fill="tozeroy",
                fillcolor=COLORS["teal_dim"],
                line=dict(color=COLORS["teal"], width=2),
            ), row=2, col=1)

        fig_hist.update_layout(
            **CHART_LAYOUT,
            margin=MARGIN_DEFAULT,
            height=560,
            showlegend=True,
        )
        fig_hist.update_xaxes(gridcolor=COLORS["border"])
        fig_hist.update_yaxes(gridcolor=COLORS["border"])
        st.plotly_chart(fig_hist, use_container_width=True)

        # Tablas raw
        with st.expander("Serie temporal completa"):
            display_serie = serie.sort_values("fecha", ascending=False).copy()
            display_serie["fecha"] = display_serie["fecha"].dt.strftime("%d %b %Y")
            st.dataframe(display_serie.reset_index(drop=True), use_container_width=True,
                         hide_index=True, height=380)

        if not ventas.empty:
            with st.expander("Transacciones individuales"):
                st.dataframe(
                    ventas.sort_values("fecha", ascending=False).reset_index(drop=True),
                    use_container_width=True, hide_index=True, height=380,
                )

        # Descarga
        st.divider()
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "Descargar serie temporal (CSV)",
                serie.to_csv(index=False).encode("utf-8"),
                "dtf_serie_temporal.csv",
                "text/csv",
            )
        with col_d2:
            if not ventas.empty:
                st.download_button(
                    "Descargar transacciones (CSV)",
                    ventas.to_csv(index=False).encode("utf-8"),
                    "dtf_ventas.csv",
                    "text/csv",
                )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — AVISOS DEL SISTEMA
# ═════════════════════════════════════════════════════════════════════════════

with tab6:
    st.header("Avisos del Sistema")
    st.markdown(timestamp_label("Estado verificado"), unsafe_allow_html=True)

    ultimo = cargar_ultimo_entrenamiento()
    serie = cargar_serie()
    db_ok = check_db()

    # ── Estado de la base de datos ──
    st.subheader("Conexión a base de datos")
    if db_ok:
        st.markdown(
            '<div class="alerta-box alerta-box-success">'
            '<span class="alerta-titulo alerta-titulo-success">Conexion activa</span>'
            'La plataforma esta conectada correctamente a la base de datos. '
            'Todos los datos estan disponibles.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="alerta-box alerta-box-danger">'
            '<span class="alerta-titulo alerta-titulo-danger">Sin conexion</span>'
            'No se puede alcanzar la base de datos. Verifica que PostgreSQL este corriendo '
            'y que las variables de entorno esten configuradas correctamente.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Estado del modelo ──
    st.subheader("Estado de los pronósticos")
    if ultimo is None:
        st.markdown(
            '<div class="alerta-box alerta-box-danger">'
            '<span class="alerta-titulo alerta-titulo-danger">Sin analisis ejecutado</span>'
            'Aun no se ha ejecutado ningun analisis de modelos. '
            'Carga tus datos de ventas y ejecuta el analisis desde el panel lateral.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        dias_sin = (datetime.now() - ultimo).days
        fecha_fmt = ultimo.strftime("%d de %B de %Y a las %H:%M")

        if dias_sin > 30:
            st.markdown(
                f'<div class="alerta-box alerta-box-danger">'
                f'<span class="alerta-titulo alerta-titulo-danger">Modelo desactualizado — {dias_sin} dias sin renovar</span>'
                f'El ultimo analisis se ejecuto el {fecha_fmt}. '
                f'Los pronosticos pueden haberse vuelto menos precisos. '
                f'Se recomienda ejecutar un nuevo analisis de modelos para mantener la exactitud.'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif dias_sin > 14:
            st.markdown(
                f'<div class="alerta-box">'
                f'<span class="alerta-titulo">Modelo proximal a vencer — {dias_sin} dias desde el ultimo analisis</span>'
                f'El ultimo analisis se ejecuto el {fecha_fmt}. '
                f'Considera ejecutar un nuevo analisis pronto para mantener la precision del pronostico.'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="alerta-box alerta-box-success">'
                f'<span class="alerta-titulo alerta-titulo-success">Modelo actualizado — {dias_sin} dias desde el ultimo analisis</span>'
                f'El ultimo analisis se ejecuto el {fecha_fmt}. '
                f'Los pronosticos estan al dia.'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Estado de los datos ──
    st.subheader("Estado de los datos")
    if serie.empty:
        st.markdown(
            '<div class="alerta-box alerta-box-danger">'
            '<span class="alerta-titulo alerta-titulo-danger">Sin datos de ventas</span>'
            'No hay datos cargados en el sistema. '
            'Sube tu archivo de ventas para comenzar a generar pronosticos.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        serie["fecha"] = pd.to_datetime(serie["fecha"])
        fecha_inicio = serie["fecha"].min().strftime("%d %b %Y")
        fecha_fin = serie["fecha"].max().strftime("%d %b %Y")
        dias_total = len(serie)
        dias_activos = (serie["unidades"] > 0).sum()

        st.markdown(
            f'<div class="alerta-box alerta-box-success">'
            f'<span class="alerta-titulo alerta-titulo-success">Datos disponibles</span>'
            f'Periodo cubierto: {fecha_inicio} al {fecha_fin} '
            f'({dias_total} dias totales, {dias_activos} con ventas registradas).'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Alerta si los datos tienen mas de 30 dias sin actualizarse
        dias_sin_datos = (datetime.now() - serie["fecha"].max()).days
        if dias_sin_datos > 30:
            st.markdown(
                f'<div class="alerta-box alerta-box-danger">'
                f'<span class="alerta-titulo alerta-titulo-danger">Datos desactualizados — {dias_sin_datos} dias sin nuevas ventas</span>'
                f'El ultimo registro de ventas es del {fecha_fin}. '
                f'Sube las ventas recientes para que los pronosticos reflejen el comportamiento actual.'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Resumen de metricas del ultimo analisis ──
    metricas_df, run_info = cargar_metricas()
    if not metricas_df.empty:
        st.subheader("Métricas del último análisis")
        col1, col2, col3 = st.columns(3)
        with col1:
            ganador = run_info.get("modelo_ganador", "N/D")
            st.metric("Modelo activo", ganador if ganador in MODELOS_VISIBLES else "N/D")
        with col2:
            mape = run_info.get("mejor_mape", None)
            st.metric("Margen de error", f"{mape:.1f}%" if mape else "N/D",
                      help="Que tan lejos esta el pronostico de las ventas reales, en promedio")
        with col3:
            mejora = run_info.get("mejora_pct", None)
            st.metric("Mejora sobre estimacion manual",
                      f"{mejora:+.1f}%" if mejora else "N/D",
                      help="Cuanto mejor pronostica la IA vs no usar ningun modelo")
