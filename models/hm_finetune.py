"""
hm_finetune.py — Fine-tuning estadístico H&M → DTF Fashion
===========================================================
Calcula índices estacionales desde el dataset H&M procesado
(hm_ventas_agregadas.csv) y los calibra a la escala real de
DTF Fashion con los datos de ventas disponibles en PostgreSQL.

Reemplaza los dicts hardcodeados en train_models.py por valores
calculados dinámicamente y persistidos en data/hm_indices.json.

El módulo se invoca en cada reentrenamiento para actualizar los
factores de corrección mensual conforme llegan nuevas ventas.

Estrategia (idéntica a research/dtf_finetuning.py de Santiago):
  1. Cargar serie diaria H&M (732 días, sep 2018 – sep 2020)
  2. Extraer índice semanal y mensual como ratio vs media global
  3. Calcular factor de escala: promedio_dtf / promedio_hm
  4. Calcular correcciones mensuales con suavizado Laplace
     (meses con < 3 observaciones DTF usan mediana como prior)
  5. Persistir en JSON con metadatos de auditoría
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("hm_finetune")

# ── Rutas ──────────────────────────────────────────────────────────────────
_BASE = Path(__file__).resolve().parent.parent / "data"
HM_CSV       = _BASE / "hm_ventas_agregadas.csv"
INDICES_JSON = _BASE / "hm_indices.json"

# ── Fallback hardcodeado ───────────────────────────────────────────────────
# Valores originales de Santiago, usados si hm_ventas_agregadas.csv
# no está disponible (p.ej. primera ejecución sin setup).
_FALLBACK: dict = {
    "indice_semanal": {
        0: 0.94, 1: 0.89, 2: 0.91, 3: 0.95,
        4: 1.05, 5: 1.16, 6: 1.10,
    },
    "indice_mensual": {
        1: 0.92, 2: 0.90, 3: 0.86, 4: 0.95, 5: 1.14, 6: 1.41,
        7: 1.16, 8: 0.95, 9: 1.03, 10: 0.93, 11: 0.93, 12: 0.84,
    },
    "correccion_mensual": {
        1: 1.100, 2: 1.529, 3: 1.754, 4: 1.200, 5: 1.000, 6: 1.000,
        7: 1.000, 8: 1.000, 9: 1.000, 10: 0.704, 11: 0.702, 12: 0.900,
    },
    "factor_escala":        0.0000346,
    "origen":               "fallback_hardcoded",
    "computed_at":          None,
    "n_observaciones_dtf":  0,
    "n_dias_hm":            732,
    "media_global_hm":      None,
}


# ══════════════════════════════════════════════════════════════════════════
# CÁLCULO DE ÍNDICES
# ══════════════════════════════════════════════════════════════════════════

def calcular_indices_hm(df_dtf: pd.DataFrame) -> dict:
    """
    Calcula índices estacionales desde hm_ventas_agregadas.csv y los
    calibra con los datos DTF reales.

    Args:
        df_dtf: DataFrame con columnas 'fecha' (date) y 'unidades' (float).
                Debe incluir días sin ventas con unidades = 0.

    Returns:
        dict con:
          - indice_semanal  {0..6: float}  — lunes=0, domingo=6
          - indice_mensual  {1..12: float}
          - correccion_mensual {1..12: float}  — con suavizado Laplace
          - factor_escala   float
          - origen          str   ('calculado_hm_csv' | 'fallback_hardcoded')
          - computed_at     str   ISO timestamp
          - n_observaciones_dtf int
          - n_dias_hm       int
          - media_global_hm float
    """
    if not HM_CSV.exists():
        log.warning(
            "hm_ventas_agregadas.csv no encontrado en %s — "
            "usando fallback hardcodeado", HM_CSV
        )
        return _FALLBACK.copy()

    log.info("Fine-tuning H&M: calculando índices desde %s", HM_CSV.name)

    # ── 1. Cargar y preparar serie H&M ────────────────────────────────────
    hm = pd.read_csv(HM_CSV, parse_dates=["fecha"])
    hm = hm.sort_values("fecha").reset_index(drop=True)
    hm["dia_semana"] = hm["fecha"].dt.dayofweek   # 0=lun, 6=dom
    hm["mes"]        = hm["fecha"].dt.month

    media_hm = hm["total_articulos"].mean()

    # ── 2. Índices estacionales H&M (ratio vs media global) ───────────────
    idx_sem = (
        hm.groupby("dia_semana")["total_articulos"].mean() / media_hm
    )
    idx_mes = (
        hm.groupby("mes")["total_articulos"].mean() / media_hm
    )

    indice_semanal = {int(k): round(float(v), 4) for k, v in idx_sem.items()}
    indice_mensual = {int(k): round(float(v), 4) for k, v in idx_mes.items()}

    log.info(
        "  H&M: %d días | media global %.1f artículos/día",
        len(hm), media_hm,
    )

    # ── 3. Preparar serie DTF ─────────────────────────────────────────────
    dtf = df_dtf[["fecha", "unidades"]].copy()
    dtf["fecha"]    = pd.to_datetime(dtf["fecha"])
    dtf["unidades"] = pd.to_numeric(dtf["unidades"], errors="coerce").fillna(0)
    dtf["mes"]      = dtf["fecha"].dt.month

    dias_activos = dtf[dtf["unidades"] > 0]
    n_obs = int(len(dias_activos))

    if n_obs == 0:
        log.warning("Sin días con ventas DTF — usando factor escala y correcciones fallback")
        return {
            **_FALLBACK,
            "indice_semanal":  indice_semanal,
            "indice_mensual":  indice_mensual,
            "origen":          "calculado_hm_csv_sin_dtf",
            "computed_at":     datetime.now().isoformat(),
            "n_observaciones_dtf": 0,
            "n_dias_hm":       len(hm),
            "media_global_hm": round(float(media_hm), 2),
        }

    # ── 4. Factor de escala H&M → DTF ─────────────────────────────────────
    promedio_dtf  = dias_activos["unidades"].mean()
    factor_escala = round(float(promedio_dtf / media_hm), 8)

    log.info(
        "  DTF: %d días con ventas | promedio %.2f uds/día | "
        "factor escala = %.8f",
        n_obs, promedio_dtf, factor_escala,
    )

    # ── 5. Correcciones mensuales con suavizado Laplace ───────────────────
    # Mitigación AMEF #12: meses con < 3 observaciones DTF usan la
    # mediana de las correcciones conocidas como prior en lugar del
    # valor puntual (evita divisiones por cero y factores extremos).
    con_ventas = dias_activos.copy()
    raw_corr: dict[int, float] = {}

    for mes, grupo in con_ventas.groupby("mes"):
        n_mes          = len(grupo)
        prom_dtf_mes   = grupo["unidades"].mean()
        hm_escalado    = indice_mensual.get(int(mes), 1.0) * promedio_dtf

        if hm_escalado <= 0:
            continue

        if n_mes >= 3:
            # Corrección directa: suficientes datos
            raw_corr[int(mes)] = prom_dtf_mes / hm_escalado
        else:
            # Suavizado Laplace: ponderar hacia 1.0 con pocos datos
            raw_corr[int(mes)] = (prom_dtf_mes + hm_escalado) / (hm_escalado * 2)

    # Prior para meses sin ningún dato DTF
    prior = float(np.median(list(raw_corr.values()))) if raw_corr else 1.0

    correccion_mensual = {
        mes: round(raw_corr.get(mes, prior), 4)
        for mes in range(1, 13)
    }

    meses_con_datos = len(raw_corr)
    log.info(
        "  Correcciones mensuales: %d meses con datos propios | "
        "prior (mediana) = %.3f",
        meses_con_datos, prior,
    )

    return {
        "indice_semanal":      indice_semanal,
        "indice_mensual":      indice_mensual,
        "correccion_mensual":  correccion_mensual,
        "factor_escala":       factor_escala,
        "origen":              "calculado_hm_csv",
        "computed_at":         datetime.now().isoformat(),
        "n_observaciones_dtf": n_obs,
        "n_dias_hm":           len(hm),
        "media_global_hm":     round(float(media_hm), 2),
        "meses_con_datos_dtf": meses_con_datos,
    }


# ══════════════════════════════════════════════════════════════════════════
# PERSISTENCIA
# ══════════════════════════════════════════════════════════════════════════

def guardar_indices(indices: dict) -> None:
    """
    Persiste los índices calculados en data/hm_indices.json.
    Las claves de los dicts internos se guardan como strings (requisito JSON).
    """
    payload = {
        **indices,
        "indice_semanal":     {str(k): v for k, v in indices["indice_semanal"].items()},
        "indice_mensual":     {str(k): v for k, v in indices["indice_mensual"].items()},
        "correccion_mensual": {str(k): v for k, v in indices["correccion_mensual"].items()},
    }
    INDICES_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Índices H&M persistidos en %s", INDICES_JSON.name)


def cargar_indices() -> dict:
    """
    Carga índices desde data/hm_indices.json.
    Si el archivo no existe o está corrupto, devuelve el fallback hardcodeado.

    Returns:
        dict con claves int en indice_semanal, indice_mensual y
        correccion_mensual (listas para uso directo con .get(int_key)).
    """
    if not INDICES_JSON.exists():
        log.info(
            "hm_indices.json no encontrado — usando fallback hardcodeado. "
            "Ejecuta un reentrenamiento para generar el archivo."
        )
        return _FALLBACK.copy()

    try:
        raw = json.loads(INDICES_JSON.read_text(encoding="utf-8"))
        return {
            **raw,
            "indice_semanal":     {int(k): v for k, v in raw["indice_semanal"].items()},
            "indice_mensual":     {int(k): v for k, v in raw["indice_mensual"].items()},
            "correccion_mensual": {int(k): v for k, v in raw["correccion_mensual"].items()},
        }
    except Exception as exc:
        log.error(
            "Error leyendo hm_indices.json: %s — usando fallback hardcodeado", exc
        )
        return _FALLBACK.copy()
