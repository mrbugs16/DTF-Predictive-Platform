"""
FastAPI Backend v4.0 — DTF Fashion Predictive Analytics Platform
Endpoints para predicciones, métricas, tendencias, recomendaciones.
Lee/escribe PostgreSQL.
"""

import os
import sys
import shutil
import tempfile
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.connection import read_sql, engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("api")

# ═══════════════════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="DTF Fashion — Predictive Analytics API",
    description="API de análisis predictivo con IA para marcas de moda DTF",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def safe_read(query: str) -> pd.DataFrame:
    """Lee de PostgreSQL con manejo de errores."""
    try:
        return read_sql(query)
    except Exception as e:
        log.error(f"Error DB: {e}")
        raise HTTPException(status_code=500, detail=f"Error de base de datos: {str(e)}")


def df_to_records(df: pd.DataFrame) -> list:
    """Convierte DataFrame a lista de dicts serializables."""
    df = df.copy()
    for col in df.select_dtypes(include=["datetime64", "datetimetz"]).columns:
        df[col] = df[col].astype(str)
    return df.where(df.notna(), None).to_dict(orient="records")


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "nombre": "DTF Fashion Predictive Analytics API",
        "version": "4.0.0",
        "endpoints": [
            "/predict", "/metrics", "/trends", "/trends/live",
            "/recommendations", "/seasonality", "/history",
            "POST /upload", "POST /train",
        ],
    }


@app.get("/health")
def health():
    try:
        df = read_sql("SELECT 1 AS ok")
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "db": db_ok, "timestamp": str(datetime.now())}


# ─── PREDICT ──────────────────────────────────────────────────────────────

@app.get("/predict")
def predict(
    modelo: Optional[str] = Query(None, description="Filtrar por modelo (SARIMA, Prophet, Random Forest, Transferencia H&M)"),
    dias: Optional[int] = Query(None, ge=1, le=90, description="Limitar horizonte"),
):
    """
    Retorna las predicciones del último training run.
    Incluye bandas de incertidumbre ±30%.
    """
    where = ""
    params = {}
    if modelo:
        where = " AND p.modelo = :modelo"
        params["modelo"] = modelo

    query = f"""
        SELECT p.*, t.modelo_ganador, t.mejora_pct
        FROM predicciones p
        LEFT JOIN training_runs t ON p.run_id = t.run_id
        WHERE 1=1 {where}
        ORDER BY p.modelo, p.fecha_prediccion
    """

    df = safe_read(query)
    if df.empty:
        raise HTTPException(status_code=404, detail="No hay predicciones. Ejecuta POST /train primero.")

    if dias:
        df = df[df["dia_horizonte"] <= dias]

    # Agrupar por modelo
    modelos = {}
    for nombre, grupo in df.groupby("modelo"):
        modelos[nombre] = {
            "predicciones": df_to_records(grupo),
            "total_unidades": round(grupo["unidades_predichas"].sum(), 1),
            "promedio_diario": round(grupo["unidades_predichas"].mean(), 2),
            "rango": {
                "inferior": round(grupo["banda_inferior"].sum(), 1),
                "superior": round(grupo["banda_superior"].sum(), 1),
            },
        }

    ganador = df["modelo_ganador"].iloc[0] if "modelo_ganador" in df.columns and df["modelo_ganador"].notna().any() else None
    mejora = df["mejora_pct"].iloc[0] if "mejora_pct" in df.columns and df["mejora_pct"].notna().any() else None

    return {
        "modelo_ganador": ganador,
        "mejora_vs_baseline_pct": mejora,
        "horizonte_dias": dias or df["dia_horizonte"].max(),
        "modelos": modelos,
    }


# ─── METRICS ──────────────────────────────────────────────────────────────

@app.get("/metrics")
def metrics():
    """
    Retorna métricas de todos los modelos del último training run,
    incluyendo baseline naive y comparación.
    """
    # Último run
    runs = safe_read("SELECT * FROM training_runs ORDER BY fecha_ejecucion DESC LIMIT 1")
    if runs.empty:
        raise HTTPException(status_code=404, detail="No hay entrenamientos registrados.")

    run = runs.iloc[0]
    run_id = run["run_id"]

    # Métricas por modelo
    metricas = safe_read(f"SELECT * FROM metricas_modelos WHERE run_id = '{run_id}' ORDER BY mape")

    return {
        "run_id": run_id,
        "fecha_ejecucion": str(run["fecha_ejecucion"]),
        "modelo_ganador": run["modelo_ganador"],
        "n_datos": int(run["n_datos"]) if pd.notna(run["n_datos"]) else None,
        "baseline": {
            "mae": float(run["baseline_mae"]) if pd.notna(run["baseline_mae"]) else None,
            "mape": float(run["baseline_mape"]) if pd.notna(run["baseline_mape"]) else None,
        },
        "mejora_pct": float(run["mejora_pct"]) if pd.notna(run["mejora_pct"]) else None,
        "modelos": df_to_records(metricas),
    }


# ─── TRENDS (datos de Google Trends desde DB) ────────────────────────────

@app.get("/trends")
def trends():
    """
    Retorna análisis de tendencias basado en patrones estacionales
    detectados en los datos y los índices H&M.
    """
    factores = safe_read("SELECT * FROM factores_hm ORDER BY tipo, clave")
    if factores.empty:
        raise HTTPException(status_code=404, detail="No hay factores H&M. Ejecuta el ETL primero.")

    # Organizar por tipo
    resultado = {}
    for tipo, grupo in factores.groupby("tipo"):
        resultado[tipo] = {
            row["clave"]: {
                "valor": float(row["valor"]),
                "descripcion": row.get("descripcion", ""),
            }
            for _, row in grupo.iterrows()
        }

    # Agregar insights
    serie = safe_read("SELECT * FROM serie_semanal ORDER BY fecha")
    insights = {}
    if not serie.empty:
        serie["fecha"] = pd.to_datetime(serie["fecha"])
        # Día más fuerte
        por_dia = serie.groupby("dia_semana")["unidades"].mean()
        mejor_dia = por_dia.idxmax()
        dias_nombre = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}
        insights["mejor_dia"] = {"dia": dias_nombre.get(mejor_dia, str(mejor_dia)), "promedio": round(por_dia.max(), 2)}

        # Mes más fuerte
        por_mes = serie.groupby("mes")["unidades"].mean()
        if por_mes.sum() > 0:
            mejor_mes = por_mes.idxmax()
            meses_nombre = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
                            7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"}
            insights["mejor_mes"] = {"mes": meses_nombre.get(mejor_mes, str(mejor_mes)), "promedio": round(por_mes.max(), 2)}

        # Tendencia reciente (últimos 30 días vs 30 anteriores)
        if len(serie) >= 60:
            reciente = serie.tail(30)["unidades"].mean()
            anterior = serie.iloc[-60:-30]["unidades"].mean()
            cambio = ((reciente - anterior) / anterior * 100) if anterior > 0 else 0
            insights["tendencia_30d"] = {"cambio_pct": round(cambio, 1), "direccion": "alza" if cambio > 0 else "baja"}

    return {
        "factores_estacionales": resultado,
        "insights": insights,
    }


# ─── TRENDS LIVE (Google Trends via pytrends) ────────────────────────────

@app.get("/trends/live")
def trends_live(
    keywords: str = Query(..., description="Keywords separados por coma (max 5)"),
    timeframe: str = Query("today 3-m", description="Timeframe: today 1-m, today 3-m, today 12-m"),
    geo: str = Query("MX", description="Código de país (MX, US, etc.)"),
):
    """
    Consulta Google Trends en tiempo real para las keywords dadas.
    Útil para detectar tendencias emergentes de diseños.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        raise HTTPException(status_code=501, detail="pytrends no instalado. pip install pytrends")

    kw_list = [k.strip() for k in keywords.split(",")][:5]
    if not kw_list:
        raise HTTPException(status_code=400, detail="Proporciona al menos una keyword")

    try:
        pytrends = TrendReq(hl="es-MX", tz=360, timeout=(10, 25))
        pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo=geo)

        # Interés en el tiempo
        interest = pytrends.interest_over_time()
        if interest.empty:
            return {"keywords": kw_list, "data": [], "mensaje": "Sin datos para estas keywords"}

        # Limpiar columna isPartial
        if "isPartial" in interest.columns:
            interest = interest.drop("isPartial", axis=1)

        interest = interest.reset_index()
        interest["date"] = interest["date"].astype(str)

        # Related queries
        related = {}
        try:
            related_queries = pytrends.related_queries()
            for kw in kw_list:
                if kw in related_queries and related_queries[kw]["top"] is not None:
                    top = related_queries[kw]["top"].head(5)
                    related[kw] = top.to_dict(orient="records")
        except Exception:
            pass

        return {
            "keywords": kw_list,
            "geo": geo,
            "timeframe": timeframe,
            "interest_over_time": interest.to_dict(orient="records"),
            "related_queries": related,
        }

    except Exception as e:
        log.error(f"Error Google Trends: {e}")
        raise HTTPException(status_code=502, detail=f"Error consultando Google Trends: {str(e)}")


# ─── RECOMMENDATIONS ─────────────────────────────────────────────────────

@app.get("/recommendations")
def recommendations():
    """
    Genera recomendaciones accionables de producción DTF
    basadas en las predicciones y patrones detectados.
    """
    # Leer predicciones del ganador
    runs = safe_read("SELECT * FROM training_runs ORDER BY fecha_ejecucion DESC LIMIT 1")
    if runs.empty:
        raise HTTPException(status_code=404, detail="No hay entrenamientos. Ejecuta POST /train.")

    ganador = runs.iloc[0]["modelo_ganador"]
    run_id = runs.iloc[0]["run_id"]

    pred = safe_read(f"""
        SELECT * FROM predicciones
        WHERE run_id = '{run_id}' AND modelo = '{ganador}'
        ORDER BY fecha_prediccion
    """)

    serie = safe_read("SELECT * FROM serie_semanal ORDER BY fecha")

    if pred.empty or serie.empty:
        raise HTTPException(status_code=404, detail="Datos insuficientes para recomendaciones.")

    pred["fecha_prediccion"] = pd.to_datetime(pred["fecha_prediccion"])
    serie["fecha"] = pd.to_datetime(serie["fecha"])

    # Análisis de ventas históricas por categoría
    ventas = safe_read("SELECT * FROM ventas")
    cat_analysis = {}
    if not ventas.empty and "categoria" in ventas.columns:
        cat_summary = ventas.groupby("categoria").agg(
            unidades=("cantidad", "sum"),
            ingreso=("ingreso_bruto", "sum"),
        ).sort_values("unidades", ascending=False)
        cat_analysis = cat_summary.to_dict(orient="index")

    # Generar recomendaciones
    total_pred = pred["unidades_predichas"].sum()
    prom_pred = pred["unidades_predichas"].mean()
    prom_historico = serie["unidades"].mean()
    cambio = ((prom_pred - prom_historico) / prom_historico * 100) if prom_historico > 0 else 0

    # Semanas del forecast
    pred["semana"] = pred["fecha_prediccion"].dt.isocalendar().week.astype(int)
    por_semana = pred.groupby("semana")["unidades_predichas"].sum()
    semana_pico = por_semana.idxmax()
    semana_baja = por_semana.idxmin()

    recomendaciones = []

    # 1. Volumen de producción
    recomendaciones.append({
        "tipo": "produccion",
        "prioridad": "alta",
        "titulo": "Volumen de producción sugerido",
        "detalle": (
            f"Prepara entre {pred['banda_inferior'].sum():.0f} y {pred['banda_superior'].sum():.0f} "
            f"unidades para los próximos 30 días (escenario central: {total_pred:.0f} uds)."
        ),
        "metrica": f"Cambio vs histórico: {cambio:+.1f}%",
    })

    # 2. Semana pico
    recomendaciones.append({
        "tipo": "timing",
        "prioridad": "alta",
        "titulo": f"Semana de mayor demanda: Semana {semana_pico}",
        "detalle": (
            f"Se estiman {por_semana[semana_pico]:.0f} unidades en la semana {semana_pico}. "
            f"Asegura inventario y capacidad de impresión DTF antes de esta semana."
        ),
    })

    # 3. Días fuertes
    pred["dia_semana"] = pred["fecha_prediccion"].dt.dayofweek
    por_dia = pred.groupby("dia_semana")["unidades_predichas"].mean()
    mejor_dia = por_dia.idxmax()
    dias_nombre = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}
    recomendaciones.append({
        "tipo": "operacion",
        "prioridad": "media",
        "titulo": f"Día más fuerte: {dias_nombre.get(mejor_dia, mejor_dia)}",
        "detalle": (
            f"El {dias_nombre.get(mejor_dia, mejor_dia)} tiene el mayor promedio esperado "
            f"({por_dia[mejor_dia]:.1f} uds). Considera concentrar lanzamientos y promociones este día."
        ),
    })

    # 4. Categorías top (si hay datos)
    if cat_analysis:
        top_cat = list(cat_analysis.keys())[:3]
        recomendaciones.append({
            "tipo": "producto",
            "prioridad": "media",
            "titulo": "Categorías prioritarias",
            "detalle": (
                f"Las categorías con mayor tracción son: {', '.join(top_cat)}. "
                f"Prioriza diseños DTF en estas líneas para los próximos 30 días."
            ),
        })

    # 5. Alerta de sobreproducción
    if cambio < -10:
        recomendaciones.append({
            "tipo": "alerta",
            "prioridad": "alta",
            "titulo": "Riesgo de sobreproducción",
            "detalle": (
                f"La demanda estimada está {abs(cambio):.0f}% por debajo del promedio histórico. "
                f"Reduce tirajes y considera producción bajo demanda."
            ),
        })
    elif cambio > 20:
        recomendaciones.append({
            "tipo": "oportunidad",
            "prioridad": "alta",
            "titulo": "Oportunidad de crecimiento",
            "detalle": (
                f"La demanda estimada supera el promedio histórico en {cambio:.0f}%. "
                f"Asegura stock de insumos DTF (film, tinta, blanks)."
            ),
        })

    return {
        "modelo_usado": ganador,
        "run_id": run_id,
        "forecast_resumen": {
            "total_unidades": round(total_pred, 1),
            "rango": [round(pred["banda_inferior"].sum(), 1), round(pred["banda_superior"].sum(), 1)],
            "cambio_vs_historico_pct": round(cambio, 1),
        },
        "recomendaciones": recomendaciones,
        "categorias": cat_analysis,
    }


# ─── SEASONALITY ─────────────────────────────────────────────────────────

@app.get("/seasonality")
def seasonality():
    """
    Retorna patrones de estacionalidad H&M vs DTF.
    Los índices H&M son calculados dinámicamente desde
    hm_ventas_agregadas.csv y calibrados con datos reales de la tienda
    (fine-tuning estadístico). Incluye metadatos de auditoría.
    """
    serie = safe_read("SELECT * FROM serie_semanal ORDER BY fecha")
    if serie.empty:
        raise HTTPException(status_code=404, detail="No hay datos de serie temporal.")

    from models.hm_finetune import cargar_indices as _cargar_hm
    hm_indices = _cargar_hm()

    HM_INDICE_SEMANAL  = hm_indices["indice_semanal"]
    HM_INDICE_MENSUAL  = hm_indices["indice_mensual"]

    dias_nombre  = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves",
                    4: "Viernes", 5: "Sábado", 6: "Domingo"}
    meses_nombre = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
                    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}

    promedio_global = serie["unidades"].mean()

    # Patrón semanal
    semanal_dtf = serie.groupby("dia_semana")["unidades"].mean()
    semanal = [
        {
            "dia":          dia,
            "nombre":       dias_nombre[dia],
            "dtf_promedio": round(float(semanal_dtf.get(dia, 0)), 2),
            "hm_escalado":  round(HM_INDICE_SEMANAL.get(dia, 1.0) * promedio_global, 2),
            "indice_hm":    round(HM_INDICE_SEMANAL.get(dia, 1.0), 4),
        }
        for dia in range(7)
    ]

    # Patrón mensual
    mensual_dtf = serie.groupby("mes")["unidades"].mean()
    mensual = [
        {
            "mes":          mes,
            "nombre":       meses_nombre[mes],
            "dtf_promedio": round(float(mensual_dtf.get(mes, 0)), 2),
            "hm_escalado":  round(HM_INDICE_MENSUAL.get(mes, 1.0) * promedio_global, 2),
            "indice_hm":    round(HM_INDICE_MENSUAL.get(mes, 1.0), 4),
        }
        for mes in range(1, 13)
    ]

    return {
        "semanal":                semanal,
        "mensual":                mensual,
        "promedio_global_diario": round(float(promedio_global), 2),
        "finetune_metadata": {
            "origen":               hm_indices.get("origen", "desconocido"),
            "computed_at":          hm_indices.get("computed_at"),
            "n_observaciones_dtf":  hm_indices.get("n_observaciones_dtf", 0),
            "n_dias_hm":            hm_indices.get("n_dias_hm", 732),
            "factor_escala":        hm_indices.get("factor_escala"),
        },
    }


# ─── HISTORY ─────────────────────────────────────────────────────────────

@app.get("/history")
def history(
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    """Retorna historial de ventas diarias."""
    df = safe_read(f"""
        SELECT fecha, unidades, ingreso_bruto, num_transacciones,
               productos_unicos, dia_nombre, es_fin_semana
        FROM serie_semanal
        ORDER BY fecha DESC
        LIMIT {limit} OFFSET {offset}
    """)

    total = safe_read("SELECT COUNT(*) as n FROM serie_semanal")
    n_total = int(total.iloc[0]["n"]) if not total.empty else 0

    return {
        "total": n_total,
        "limit": limit,
        "offset": offset,
        "data": df_to_records(df),
    }


# ─── UPLOAD ──────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Recibe un archivo Excel/CSV del usuario, ejecuta el pipeline ETL
    y retorna resumen de los datos procesados.
    """
    # Validar extensión
    ext = Path(file.filename).suffix.lower()
    if ext not in (".xlsx", ".xls", ".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado: {ext}. Usa .xlsx, .xls o .csv",
        )

    # Guardar temporalmente
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    try:
        with open(tmp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        log.info(f"Archivo recibido: {file.filename} ({len(content)} bytes)")

        # Ejecutar ETL
        from etl.etl_pipeline import ejecutar_pipeline
        resultado = ejecutar_pipeline(str(tmp_path))

        return resultado

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"Error en upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─── TRAIN ───────────────────────────────────────────────────────────────

@app.post("/train")
def train():
    """
    Ejecuta el pipeline de entrenamiento:
    SARIMA + Prophet + Random Forest + Transferencia H&M.
    Compara, selecciona ganador, guarda todo a PostgreSQL.
    """
    try:
        from models.train_models import ejecutar_entrenamiento
        resultado = ejecutar_entrenamiento()
        return resultado
    except Exception as e:
        log.error(f"Error en entrenamiento: {e}")
        raise HTTPException(status_code=500, detail=f"Error durante entrenamiento: {str(e)}")


# ─── TRAINING RUNS HISTORY ──────────────────────────────────────────────

@app.get("/runs")
def training_runs(limit: int = Query(10, ge=1, le=100)):
    """Retorna historial de training runs."""
    df = safe_read(f"""
        SELECT * FROM training_runs
        ORDER BY fecha_ejecucion DESC
        LIMIT {limit}
    """)
    return {"runs": df_to_records(df)}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )