"""
services/viability_score.py — v2: Reenmarcado como "Índice de Prioridad"
═══════════════════════════════════════════════════════════════════════════════
Diferencias vs v1:
  • Ya NO se llama "Probabilidad de éxito" — ahora es "Índice de Prioridad"
  • Lenguaje más humilde y defendible académicamente
  • Mismos cálculos numéricos (compatible con el frontend actual)
  • Interpretaciones ajustadas para no prometer predicción

Por qué este cambio:
  La versión anterior sugería una "probabilidad" (término estadístico con
  requisitos específicos: validación empírica, bandas de confianza, etc.).
  Sin datos de ground truth de viabilidad, "probabilidad" es un overclaim.
  
  "Índice de Prioridad" es honesto: es una puntuación heurística que ordena
  diseños por cuál amerita más atención, sin prometer predecir ventas.

Este archivo reemplaza completamente services/viability_score.py v1.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.connection import read_sql

log = logging.getLogger("viability_score")

# ═══════════════════════════════════════════════════════════════════════════
# PESOS — justificables académicamente (ver tesis sección X.Y)
# ═══════════════════════════════════════════════════════════════════════════

PESO_ML = 0.50       # Mayor peso a la señal con validación empírica (MAPE 15.48%)
PESO_TRENDS = 0.30   # Señal externa secundaria, sujeta a ruido
PESO_HM = 0.20       # Proxy aspiracional de la industria global

# Umbrales de interpretación — ahora hablan de PRIORIDAD, no probabilidad
UMBRAL_ALTA = 70
UMBRAL_MEDIA = 40


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENTE 1: DEMANDA HISTÓRICA DE LA CATEGORÍA
# ═══════════════════════════════════════════════════════════════════════════

def _score_demanda_ml(categoria: str) -> dict:
    """
    Mide cuánto ha vendido históricamente la categoría propuesta
    vs la categoría líder del catálogo. Escala 0-100.
    """
    try:
        ventas_query = """
            SELECT categoria, SUM(cantidad) AS total
            FROM ventas
            GROUP BY categoria
            ORDER BY total DESC
        """
        ventas = read_sql(ventas_query)
        if ventas.empty:
            return {
                "score": 50.0,
                "fuente": "default",
                "detalle": "Sin datos históricos disponibles",
            }

        max_v = ventas["total"].max()
        cat_v = ventas[ventas["categoria"] == categoria]["total"].sum()
        lider = ventas.iloc[0]["categoria"]

        if max_v > 0:
            score = (cat_v / max_v) * 100
            return {
                "score": round(float(score), 1),
                "fuente": "ml_historico",
                "detalle": (
                    f"{int(cat_v)} piezas vendidas en '{categoria}' "
                    f"vs {int(max_v)} de '{lider}' (categoría líder)"
                ),
                "categoria_lider": lider,
                "piezas_categoria": int(cat_v),
                "piezas_lider": int(max_v),
            }
    except Exception as e:
        log.warning(f"Error calculando histórico: {e}")

    return {
        "score": 50.0,
        "fuente": "default",
        "detalle": "Categoría sin registro en el historial",
    }


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENTE 2: TENDENCIA GOOGLE TRENDS
# ═══════════════════════════════════════════════════════════════════════════

def _score_trends(keywords: list, geo: str = "MX") -> dict:
    """
    Mide interés público actual y pendiente reciente para las keywords.
    Combinado: 70% interés promedio + 30% bonus por tendencia ascendente.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        return {
            "score": 50.0,
            "fuente": "default",
            "detalle": "Google Trends no disponible",
        }

    if not keywords:
        return {"score": 50.0, "fuente": "default", "detalle": "Sin términos de búsqueda"}

    kw_clean = [k.strip() for k in keywords if k and k.strip()][:5]
    if not kw_clean:
        return {"score": 50.0, "fuente": "default", "detalle": "Términos vacíos"}

    try:
        pytrends = TrendReq(hl="es-MX", tz=360, timeout=(10, 25))
        pytrends.build_payload(kw_clean, cat=0, timeframe="today 3-m", geo=geo)
        interest = pytrends.interest_over_time()

        if interest.empty:
            return {
                "score": 30.0,
                "fuente": "trends_vacio",
                "detalle": "Sin datos en Google Trends — términos muy nicho",
                "keywords": kw_clean,
            }

        if "isPartial" in interest.columns:
            interest = interest.drop("isPartial", axis=1)

        interes_promedio = float(interest.mean(axis=1).mean())

        # Pendiente: segunda mitad vs primera mitad
        n = len(interest)
        if n >= 4:
            mitad = n // 2
            primera = interest.iloc[:mitad].mean(axis=1).mean()
            segunda = interest.iloc[mitad:].mean(axis=1).mean()
            pendiente = (segunda - primera) / max(primera, 1) * 100
        else:
            pendiente = 0.0

        bonus = max(-20, min(20, pendiente))
        score = 0.7 * interes_promedio + 0.3 * (50 + bonus)
        score = max(0, min(100, score))

        return {
            "score": round(float(score), 1),
            "fuente": "google_trends",
            "detalle": (
                f"Interés promedio: {interes_promedio:.0f}/100, "
                f"tendencia reciente: {pendiente:+.1f}%"
            ),
            "keywords": kw_clean,
            "interes_promedio": round(interes_promedio, 1),
            "pendiente_pct": round(float(pendiente), 1),
        }

    except Exception as e:
        log.warning(f"Error Google Trends: {e}")
        return {
            "score": 50.0,
            "fuente": "error",
            "detalle": f"Error consultando Trends: {str(e)[:80]}",
        }


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENTE 3: ESTACIONALIDAD DE LA INDUSTRIA
# ═══════════════════════════════════════════════════════════════════════════

def _score_estacionalidad_hm(mes: Optional[int] = None) -> dict:
    """Usa el índice estacional H&M del mes actual (o pedido)."""
    mes = mes or datetime.now().month

    try:
        df = read_sql(
            f"SELECT valor FROM factores_hm WHERE tipo='mensual' AND clave='{mes}'"
        )
        if df.empty:
            return {"score": 50.0, "fuente": "default", "detalle": "Sin patrón estacional"}

        indice = float(df.iloc[0]["valor"])
        score = 50 + ((indice - 1.0) * 100)
        score = max(0, min(100, score))

        return {
            "score": round(float(score), 1),
            "fuente": "hm_estacional",
            "detalle": (
                f"Patrones estacionales de la industria — "
                f"mes {mes}: índice {indice:.2f}"
            ),
            "mes_consultado": mes,
        }
    except Exception as e:
        log.warning(f"Error leyendo factores estacionales: {e}")
        return {"score": 50.0, "fuente": "default", "detalle": str(e)[:80]}


# ═══════════════════════════════════════════════════════════════════════════
# INTERPRETACIÓN — REENMARCADA COMO PRIORIDAD, NO PROBABILIDAD
# ═══════════════════════════════════════════════════════════════════════════

def _interpretar_score(score: float) -> dict:
    """
    Traduce el score a una recomendación de prioridad de producción.
    
    CAMBIO CLAVE vs v1: ya no hablamos de "alta probabilidad de éxito".
    Ahora hablamos de "alta prioridad basada en señales convergentes".
    Esto es metodológicamente más honesto sin requerir validación empírica
    de probabilidad real.
    """
    if score >= UMBRAL_ALTA:
        return {
            "nivel": "ALTA",
            "color": "verde",
            "recomendacion": (
                "DISEÑO PRIORIZADO — las tres señales analizadas "
                "(historial, tendencias, estacionalidad) sugieren "
                "condiciones favorables para este tipo de diseño."
            ),
            "accion_sugerida": (
                "Considerar producción de tiraje amplio (50-100 piezas) "
                "tras revisar el desglose por componente."
            ),
            "disclaimer": (
                "Este índice refleja la convergencia de señales disponibles, "
                "no una predicción cuantitativa de ventas."
            ),
        }
    elif score >= UMBRAL_MEDIA:
        return {
            "nivel": "MEDIA",
            "color": "amarillo",
            "recomendacion": (
                "DISEÑO EN ZONA INTERMEDIA — las señales analizadas son "
                "mixtas. Conviene revisar el desglose para entender "
                "cuál señal tira hacia arriba y cuál hacia abajo."
            ),
            "accion_sugerida": (
                "Considerar edición limitada (15-30 piezas) y medir la "
                "respuesta del mercado antes de comprometer más producción."
            ),
            "disclaimer": (
                "Un índice intermedio no es un rechazo; es una invitación "
                "a revisar el análisis por componente."
            ),
        }
    else:
        return {
            "nivel": "BAJA",
            "color": "rojo",
            "recomendacion": (
                "DISEÑO DE BAJA PRIORIDAD — las señales analizadas "
                "sugieren condiciones desfavorables en este momento."
            ),
            "accion_sugerida": (
                "Considerar reservar este diseño para una temporada "
                "más favorable o revisar el concepto."
            ),
            "disclaimer": (
                "Un índice bajo no garantiza fracaso. Refleja únicamente "
                "la convergencia de señales en el momento del análisis."
            ),
        }


def calcular_viabilidad(
    categoria: str,
    keywords: list,
    geo: str = "MX",
    mes: Optional[int] = None,
) -> dict:
    """
    Calcula el Índice de Prioridad combinando las 3 señales.
    
    Returns:
        dict con estructura:
        {
            "score_total": 67.3,
            "interpretacion": {...},
            "componentes": {...},
            "pesos": {...},
            "metodologia": "heurística compuesta 50/30/20",
            "es_prediccion": False  # NUEVO: explícito que no es predicción
        }
    """
    log.info(f"Calculando índice: cat={categoria}, kws={keywords}")

    comp_ml = _score_demanda_ml(categoria)
    comp_trends = _score_trends(keywords, geo=geo)
    comp_hm = _score_estacionalidad_hm(mes=mes)

    score_total = (
        PESO_ML * comp_ml["score"]
        + PESO_TRENDS * comp_trends["score"]
        + PESO_HM * comp_hm["score"]
    )
    score_total = round(float(score_total), 1)

    return {
        "score_total": score_total,
        "interpretacion": _interpretar_score(score_total),
        "componentes": {
            "demanda_ml": comp_ml,
            "tendencia_trends": comp_trends,
            "estacionalidad_hm": comp_hm,
        },
        "pesos": {
            "ml": PESO_ML,
            "trends": PESO_TRENDS,
            "hm": PESO_HM,
        },
        "metodologia": "heurística compuesta 50% historial / 30% tendencias / 20% estacionalidad",
        "es_prediccion": False,   # Explícito: NO es probabilidad predictiva
        "tipo_indice": "Índice de Prioridad (heurístico)",
        "categoria_evaluada": categoria,
        "fecha_calculo": datetime.now().isoformat(),
    }