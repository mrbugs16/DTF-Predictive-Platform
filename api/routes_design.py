"""
api/routes_theme.py — Endpoint POST /api/v1/analyze-theme
═══════════════════════════════════════════════════════════════════════════════
"Explorador de Temas" — a diferencia del Analizador de Diseños (input imagen),
este endpoint recibe un TEMA textual (ej. "Copa del Mundo 2026", "Halloween")
y responde con:

  1. Análisis de Google Trends del tema en el geo especificado
  2. Categoría DTF más afín al tema (estimada heurísticamente o forzada)
  3. PRONÓSTICO REAL del modelo ML ganador para los próximos 30 días
  4. Recomendación cuantitativa en piezas

Diferencia fundamental vs el Analizador de Diseños:
  • Analizador de Diseños: score heurístico 0-100 (Índice de Prioridad)
  • Explorador de Temas: PREDICCIÓN EN PIEZAS del modelo SARIMA/Prophet/RF

Esto es más defendible porque usa los modelos entrenados, no heurísticas.

Para integrarlo en api/main.py, agrega:
    from api.routes_theme import router as theme_router
    app.include_router(theme_router)
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import os

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.connection import read_sql

from services.design_vision import analyze_design
from services.viability_score import calcular_viabilidad

log = logging.getLogger("api.theme")

router = APIRouter(prefix="/api/v1", tags=["Theme Explorer"])


# Categorías válidas — mismas que design_vision.py
CATEGORIAS_VALIDAS = [
    "Sports", "Gym", "Futbol", "Basketball", "Tenis", "Casual",
    "Movies", "Musica", "Hockey", "Deportiva", "Skateboarding",
    "Baseball", "Ufc", "Track and Field",
]

# Mapeo keyword → categoría DTF — heurístico pero explícito
# Orden importa: más específicos primero
MAPEO_KEYWORDS_CATEGORIA = [
    (["futbol", "fútbol", "copa del mundo", "mundial", "liga mx", "chivas", "america", "messi", "cristiano"], "Futbol"),
    (["basketball", "basquetbol", "básquetbol", "nba", "lebron"], "Basketball"),
    (["tenis", "tennis", "nadal", "federer", "djokovic"], "Tenis"),
    (["baseball", "béisbol", "beisbol", "mlb", "yankees"], "Baseball"),
    (["hockey", "nhl"], "Hockey"),
    (["ufc", "mma", "boxeo", "pelea"], "Ufc"),
    (["skate", "skateboard", "skateboarding", "patineta"], "Skateboarding"),
    (["track", "atletismo", "correr", "running", "maraton", "maratón"], "Track and Field"),
    (["gym", "fitness", "crossfit", "pesas", "musculacion"], "Gym"),
    (["deporte", "deportiva", "atleta", "entrenamiento"], "Deportiva"),
    (["sport", "sports"], "Sports"),
    (["pelicula", "película", "movie", "cine", "marvel", "dc", "disney"], "Movies"),
    (["musica", "música", "music", "concierto", "banda", "rock", "pop", "reggaeton"], "Musica"),
]


# ═══════════════════════════════════════════════════════════════════════════
# MODELOS Pydantic
# ═══════════════════════════════════════════════════════════════════════════

class AnalyzeThemeRequest(BaseModel):
    """Request body para POST /analyze-theme."""
    tema: str = Field(..., min_length=2, max_length=100,
                      description="Tema a explorar (ej. 'Copa del Mundo 2026', 'Halloween')")
    geo: str = Field("MX", description="Código de país para Google Trends")
    categoria_override: Optional[str] = Field(
        None, description=f"Forzar categoría. Opciones: {', '.join(CATEGORIAS_VALIDAS)}")
    horizonte_dias: int = Field(30, ge=7, le=90, description="Horizonte de pronóstico en días")


class ThemeAnalysis(BaseModel):
    interes_promedio: float = Field(..., ge=0, le=100)
    interes_maximo: float
    pendiente_pct: float
    clasificacion: str = Field(..., description="CRECIENTE | ESTABLE | DECRECIENTE")


class CategoryMatch(BaseModel):
    categoria_detectada: str
    metodo: str = Field(..., description="keyword_match | override_manual | default")
    confianza: str = Field(..., description="alta | media | baja")


class ForecastResult(BaseModel):
    modelo_usado: str
    piezas_estimadas_horizonte: float
    piezas_estimadas_banda_inferior: float
    piezas_estimadas_banda_superior: float
    promedio_diario: float
    horizonte_dias: int


class AnalyzeThemeResponse(BaseModel):
    tema_analizado: str
    analisis_trends: ThemeAnalysis
    category_match: CategoryMatch
    forecast: Optional[ForecastResult] = None
    recomendacion: dict
    serie_trends: list = Field(default_factory=list,
                                description="Puntos [fecha, interes] para graficar")
    metadata: dict


# ═══════════════════════════════════════════════════════════════════════════
# DETECCIÓN DE CATEGORÍA POR KEYWORDS
# ═══════════════════════════════════════════════════════════════════════════

def detectar_categoria(tema: str) -> dict:
    """
    Detecta la categoría DTF más afín al tema usando keyword matching.
    Si no encuentra match, devuelve 'Casual' con confianza baja.
    """
    tema_lower = tema.lower()

    for keywords, categoria in MAPEO_KEYWORDS_CATEGORIA:
        for kw in keywords:
            if kw in tema_lower:
                return {
                    "categoria_detectada": categoria,
                    "metodo": "keyword_match",
                    "confianza": "alta" if len(kw) > 4 else "media",
                    "keyword_matched": kw,
                }

    return {
        "categoria_detectada": "Casual",
        "metodo": "default",
        "confianza": "baja",
        "keyword_matched": None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CONSULTA A GOOGLE TRENDS
# ═══════════════════════════════════════════════════════════════════════════

def consultar_trends(tema: str, geo: str = "MX") -> dict:
    """
    Consulta Google Trends para el tema y devuelve serie temporal + métricas.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        raise HTTPException(status_code=501, detail="pytrends no instalado")

    try:
        pytrends = TrendReq(hl="es-MX", tz=360, timeout=(10, 25))
        pytrends.build_payload([tema], cat=0, timeframe="today 12-m", geo=geo)
        interest = pytrends.interest_over_time()

        if interest.empty:
            return {
                "error": "Sin datos en Google Trends",
                "interes_promedio": 0,
                "interes_maximo": 0,
                "pendiente_pct": 0,
                "clasificacion": "SIN_DATOS",
                "serie": [],
            }

        if "isPartial" in interest.columns:
            interest = interest.drop("isPartial", axis=1)

        serie_col = interest[tema]

        # Métricas
        interes_promedio = float(serie_col.mean())
        interes_maximo = float(serie_col.max())

        # Pendiente: últimos 3 meses vs anteriores
        n = len(serie_col)
        if n >= 12:
            reciente = serie_col.iloc[-13:].mean()  # ~3 meses
            anterior = serie_col.iloc[:-13].mean()  # meses previos
            pendiente = (reciente - anterior) / max(anterior, 1) * 100
        else:
            pendiente = 0.0

        # Clasificación
        if pendiente > 15:
            clasif = "CRECIENTE"
        elif pendiente < -15:
            clasif = "DECRECIENTE"
        else:
            clasif = "ESTABLE"

        # Serie para graficar (formato JSON-safe)
        serie = [
            {"fecha": str(idx.date()), "interes": int(val)}
            for idx, val in serie_col.items()
        ]

        return {
            "interes_promedio": round(interes_promedio, 1),
            "interes_maximo": round(interes_maximo, 1),
            "pendiente_pct": round(float(pendiente), 1),
            "clasificacion": clasif,
            "serie": serie,
        }

    except Exception as e:
        log.exception("Error Google Trends")
        return {
            "error": str(e),
            "interes_promedio": 0,
            "interes_maximo": 0,
            "pendiente_pct": 0,
            "clasificacion": "ERROR",
            "serie": [],
        }


# ═══════════════════════════════════════════════════════════════════════════
# CONSULTA A MODELOS ML (PREDICCIÓN REAL)
# ═══════════════════════════════════════════════════════════════════════════

def obtener_forecast_modelo(horizonte_dias: int = 30) -> Optional[dict]:
    """
    Lee la predicción más reciente del modelo ganador desde la BD.
    Devuelve piezas estimadas en el horizonte.

    Esto es lo que hace este endpoint ACADÉMICAMENTE DEFENDIBLE:
    usa los modelos entrenados, no heurísticas.
    """
    try:
        # Última corrida
        runs = read_sql(
            "SELECT * FROM training_runs ORDER BY fecha_ejecucion DESC LIMIT 1"
        )
        if runs.empty:
            return None

        run = runs.iloc[0]
        ganador = run["modelo_ganador"]
        run_id = run["run_id"]

        # Predicciones del modelo ganador
        pred = read_sql(f"""
            SELECT * FROM predicciones
            WHERE run_id = '{run_id}' AND modelo = '{ganador}'
              AND dia_horizonte <= {horizonte_dias}
            ORDER BY fecha_prediccion
        """)

        if pred.empty:
            return None

        return {
            "modelo_usado": ganador,
            "piezas_estimadas_horizonte": round(float(pred["unidades_predichas"].sum()), 1),
            "piezas_estimadas_banda_inferior": round(float(pred["banda_inferior"].sum()), 1),
            "piezas_estimadas_banda_superior": round(float(pred["banda_superior"].sum()), 1),
            "promedio_diario": round(float(pred["unidades_predichas"].mean()), 2),
            "horizonte_dias": horizonte_dias,
        }

    except Exception as e:
        log.warning(f"No se pudo leer forecast: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# GENERACIÓN DE RECOMENDACIÓN CUANTITATIVA
# ═══════════════════════════════════════════════════════════════════════════

def generar_recomendacion(trends: dict, forecast: Optional[dict],
                           categoria_match: dict) -> dict:
    """
    Genera recomendación combinando Trends + forecast.
    
    A diferencia del score heurístico del Analizador de Diseños,
    aquí la recomendación se expresa en PIEZAS REALES según el modelo
    entrenado, moduladas por el nivel de interés del tema.
    """
    if forecast is None:
        return {
            "estado": "SIN_FORECAST",
            "mensaje": (
                "No se puede generar recomendación cuantitativa porque no hay "
                "modelo entrenado. Ejecuta el análisis de modelos en el panel lateral."
            ),
            "accion_sugerida": "Entrenar modelos antes de evaluar temas.",
        }

    interes = trends.get("interes_promedio", 0)
    pendiente = trends.get("pendiente_pct", 0)
    clasif = trends.get("clasificacion", "ESTABLE")

    # Modulador basado en Trends: tema caliente amplifica el forecast
    if interes >= 60 and clasif == "CRECIENTE":
        modulador = 1.3
        nivel = "ALTA"
    elif interes >= 40 or clasif == "CRECIENTE":
        modulador = 1.0
        nivel = "MEDIA"
    elif interes >= 20 and clasif == "ESTABLE":
        modulador = 0.85
        nivel = "MEDIA"
    else:
        modulador = 0.6
        nivel = "BAJA"

    # Proyección ajustada
    piezas_base = forecast["piezas_estimadas_horizonte"]
    piezas_ajustadas = piezas_base * modulador
    piezas_banda_baja = forecast["piezas_estimadas_banda_inferior"] * modulador
    piezas_banda_alta = forecast["piezas_estimadas_banda_superior"] * modulador

    # Confianza del match de categoría afecta el rango
    if categoria_match["confianza"] == "baja":
        # Ampliamos la banda si no estamos seguros de la categoría
        piezas_banda_baja *= 0.7
        piezas_banda_alta *= 1.3

    # Mensajes según nivel
    mensajes = {
        "ALTA": {
            "titulo": "Tema con alto potencial",
            "mensaje": (
                f"El tema '{clasif.lower()}' con interés promedio {interes:.0f}/100 en Google. "
                f"Combinado con el modelo {forecast['modelo_usado']}, sugiere producir "
                f"entre {piezas_banda_baja:.0f} y {piezas_banda_alta:.0f} piezas "
                f"en los próximos {forecast['horizonte_dias']} días."
            ),
            "accion": (
                f"Considerar tiraje agresivo: {piezas_ajustadas:.0f} piezas "
                f"(rango {piezas_banda_baja:.0f}–{piezas_banda_alta:.0f})."
            ),
        },
        "MEDIA": {
            "titulo": "Tema con potencial moderado",
            "mensaje": (
                f"Interés {interes:.0f}/100 con tendencia {clasif.lower()}. "
                f"El modelo {forecast['modelo_usado']} proyecta "
                f"{piezas_ajustadas:.0f} piezas en {forecast['horizonte_dias']} días "
                f"(rango {piezas_banda_baja:.0f}–{piezas_banda_alta:.0f})."
            ),
            "accion": (
                f"Tiraje conservador: {piezas_banda_baja:.0f}–{piezas_ajustadas:.0f} piezas, "
                "con opción de reimprimir si la respuesta es positiva."
            ),
        },
        "BAJA": {
            "titulo": "Tema con bajo potencial en este momento",
            "mensaje": (
                f"Interés bajo ({interes:.0f}/100) o tendencia {clasif.lower()}. "
                f"El modelo proyecta solo {piezas_ajustadas:.0f} piezas estimadas "
                f"si se lanzara ahora."
            ),
            "accion": (
                "Considerar esperar una temporada más favorable o redirigir "
                "el concepto hacia una categoría más demandada."
            ),
        },
    }

    info = mensajes[nivel]

    return {
        "estado": "OK",
        "nivel": nivel,
        "titulo": info["titulo"],
        "mensaje": info["mensaje"],
        "accion_sugerida": info["accion"],
        "piezas_estimadas_central": round(float(piezas_ajustadas), 0),
        "piezas_estimadas_rango": {
            "minimo": round(float(piezas_banda_baja), 0),
            "maximo": round(float(piezas_banda_alta), 0),
        },
        "modulador_aplicado": modulador,
        "nota_confianza": (
            "Rango ampliado por baja confianza en detección de categoría."
            if categoria_match["confianza"] == "baja"
            else None
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/analyze-theme",
    response_model=AnalyzeThemeResponse,
    summary="Explorador de Temas — evalúa la oportunidad de merch sobre un tema",
    description="""
Recibe un tema textual (ej. "Copa del Mundo 2026") y devuelve:

1. **Análisis de Google Trends** del tema en el geo especificado (12 meses)
2. **Categoría DTF más afín** detectada por keyword matching
3. **Pronóstico real** del modelo ML ganador (SARIMA/Prophet/RF) para N días
4. **Recomendación cuantitativa en piezas** modulada por el interés del tema

A diferencia del Analizador de Diseños (heurística sobre imagen), este
endpoint usa los modelos ML entrenados sobre 31.7M transacciones H&M
calibrados a la escala DTF, dando predicciones cuantitativas defendibles.
""",
)
async def analyze_theme_endpoint(request: AnalyzeThemeRequest):
    log.info(f"📥 Analizando tema: '{request.tema}' (geo={request.geo})")

    # 1. Detectar categoría
    if request.categoria_override:
        if request.categoria_override not in CATEGORIAS_VALIDAS:
            raise HTTPException(
                status_code=400,
                detail=f"Categoría inválida: {request.categoria_override}",
            )
        category_match = {
            "categoria_detectada": request.categoria_override,
            "metodo": "override_manual",
            "confianza": "alta",
        }
    else:
        category_match = detectar_categoria(request.tema)

    log.info(f"🏷️  Categoría: {category_match['categoria_detectada']} "
             f"({category_match['metodo']}, confianza {category_match['confianza']})")

    # 2. Consultar Google Trends
    trends = consultar_trends(request.tema, geo=request.geo)
    if "error" in trends:
        log.warning(f"Trends error: {trends['error']}")

    log.info(f"📊 Trends: interés {trends.get('interes_promedio', 0):.0f}/100, "
             f"pendiente {trends.get('pendiente_pct', 0):+.1f}%")

    # 3. Forecast del modelo ganador
    forecast = obtener_forecast_modelo(horizonte_dias=request.horizonte_dias)

    if forecast:
        log.info(f"🤖 Forecast ({forecast['modelo_usado']}): "
                 f"{forecast['piezas_estimadas_horizonte']:.0f} pzas en "
                 f"{forecast['horizonte_dias']} días")
    else:
        log.warning("Sin forecast disponible")

    # 4. Recomendación
    recomendacion = generar_recomendacion(trends, forecast, category_match)

    # Respuesta estructurada
    return AnalyzeThemeResponse(
        tema_analizado=request.tema,
        analisis_trends=ThemeAnalysis(
            interes_promedio=trends.get("interes_promedio", 0),
            interes_maximo=trends.get("interes_maximo", 0),
            pendiente_pct=trends.get("pendiente_pct", 0),
            clasificacion=trends.get("clasificacion", "SIN_DATOS"),
        ),
        category_match=CategoryMatch(**{
            k: v for k, v in category_match.items()
            if k in ["categoria_detectada", "metodo", "confianza"]
        }),
        forecast=ForecastResult(**forecast) if forecast else None,
        recomendacion=recomendacion,
        serie_trends=trends.get("serie", []),
        metadata={
            "geo": request.geo,
            "horizonte_dias": request.horizonte_dias,
            "fecha_analisis": datetime.now().isoformat(),
            "usa_modelos_ml": forecast is not None,
            "trends_error": trends.get("error"),
        },
    )

# ═════════════════════════════════════════════════════════════════
# Endpoint de análisis de imagen
# ═════════════════════════════════════════════════════════════════

@router.post("/analyze-design", summary="Analizar diseño de imagen DTF")
async def analyze_design_endpoint(
    file: UploadFile = File(...),
    geo: str = Form("MX"),
    categoria_override: Optional[str] = Form(None)
):
    """
    Recibe una imagen de un diseño DTF, extrae keywords, categoría,
    y calcula el Índice de Prioridad con ML + Google Trends + estacionalidad H&M.
    """
    # Validar imagen
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    image_bytes = await file.read()

    # 1. Análisis visual (OpenAI/Claude/Mock)
    try:
        analisis_visual = analyze_design(image_bytes, provider=os.getenv("VISION_PROVIDER"))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error en análisis visual: {str(e)}")

    categoria_final = categoria_override or analisis_visual.get("categoria_sugerida", "Casual")
    keywords = analisis_visual.get("keywords", [])

    # 2. Índice de prioridad (viabilidad)
    try:
        score_data = calcular_viabilidad(categoria_final, keywords, geo=geo)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculando viabilidad: {str(e)}")

    return {
        "score_viabilidad": score_data["score_total"],
        "interpretacion": score_data["interpretacion"],
        "componentes_score": score_data["componentes"],
        "pesos": score_data["pesos"], 
        "analisis_visual": analisis_visual,
        "categoria_usada": categoria_final,
        "keywords_usadas": keywords,
    }