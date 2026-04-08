"""
Train Models v4.1 — DTF Fashion Predictive Analytics Platform
Entrena SARIMA, Prophet, Random Forest.
Compara automáticamente, selecciona ganador, guarda a PostgreSQL.

Los índices estacionales de retail (derivados de H&M) se inyectan como
features del Random Forest vía la tabla 'features' del ETL. No se exponen
al usuario como modelo independiente.
"""

import os
import sys
import logging
import warnings
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.connection import engine, read_sql, write_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("train_models")

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════

HORIZONTE_DIAS = 30
BANDA_INCERTIDUMBRE = 0.30  # ±30%
MIN_DATOS_SARIMA = 14       # mínimo para SARIMA
MIN_DATOS_RF = 30           # mínimo para Random Forest (necesita lags)

# Índices H&M — se cargan desde data/hm_indices.json (generado por
# hm_finetune.py en cada reentrenamiento). Si el archivo no existe
# todavía, cargar_indices() devuelve el fallback hardcodeado original
# para mantener compatibilidad sin necesidad de setup previo.
from models.hm_finetune import (
    calcular_indices_hm,
    guardar_indices,
    cargar_indices as _cargar_indices_hm,
)

_hm_cargado       = _cargar_indices_hm()
HM_INDICE_SEMANAL     = _hm_cargado["indice_semanal"]
HM_INDICE_MENSUAL     = _hm_cargado["indice_mensual"]
DTF_CORRECCION_MENSUAL = _hm_cargado["correccion_mensual"]

log.info(
    "Índices H&M cargados — origen: %s | computed_at: %s",
    _hm_cargado.get("origen", "?"),
    _hm_cargado.get("computed_at", "no disponible"),
)


# ═══════════════════════════════════════════════════════════════════════════
# MÉTRICAS DE EVALUACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula MAE, MAPE y R² entre valores reales y predichos."""
    y_real = np.array(y_real, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = np.mean(np.abs(y_real - y_pred))

    # MAPE: evitar división por cero
    mask = y_real != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100
    else:
        mape = 100.0

    # R²
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mae": round(mae, 4),
        "mape": round(mape, 2),
        "r2": round(r2, 4),
    }


def calcular_baseline_naive(serie: pd.Series, horizonte: int) -> dict:
    """
    Baseline naive: repetir los últimos `horizonte` días como pronóstico.
    Sirve como referencia para medir mejora ≥20-25%.
    """
    if len(serie) < horizonte * 2:
        # Si no hay suficientes datos, usar la media
        pred = np.full(horizonte, serie.mean())
        real = serie.values[-horizonte:]
        if len(real) < horizonte:
            real = np.pad(real, (0, horizonte - len(real)), constant_values=serie.mean())
    else:
        real = serie.values[-horizonte:]
        pred = serie.values[-2 * horizonte : -horizonte]

    return calcular_metricas(real, pred)


# ═══════════════════════════════════════════════════════════════════════════
# MODELO 1: SARIMA
# ═══════════════════════════════════════════════════════════════════════════

def entrenar_sarima(serie: pd.Series, horizonte: int = HORIZONTE_DIAS) -> dict:
    """
    Entrena SARIMA(1,1,1)(1,1,1)[7] sobre la serie de unidades diarias.
    """
    log.info("─── Entrenando SARIMA(1,1,1)(1,1,1)[7] ───")
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    n = len(serie)
    if n < MIN_DATOS_SARIMA:
        log.warning(f"  → Solo {n} datos. SARIMA necesita al menos {MIN_DATOS_SARIMA}.")
        return {"status": "skip", "razon": "datos insuficientes"}

    # Split train/test
    test_size = min(horizonte, max(7, n // 5))
    train = serie.iloc[:-test_size]
    test = serie.iloc[-test_size:]

    try:
        modelo = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        resultado = modelo.fit(disp=False, maxiter=200)

        # Predicción sobre test
        pred_test = resultado.forecast(steps=test_size)
        pred_test = np.maximum(pred_test, 0)  # No permitir negativos

        metricas = calcular_metricas(test.values, pred_test.values)

        # Reentrenar con toda la serie para forecast futuro
        modelo_full = SARIMAX(
            serie,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        resultado_full = modelo_full.fit(disp=False, maxiter=200)

        forecast = resultado_full.forecast(steps=horizonte)
        forecast = np.maximum(forecast, 0)

        log.info(f"  → MAE: {metricas['mae']:.2f} | MAPE: {metricas['mape']:.1f}% | R²: {metricas['r2']:.3f}")

        return {
            "status": "ok",
            "nombre": "SARIMA",
            "metricas": metricas,
            "forecast": forecast.values,
            "pred_test": pred_test.values,
            "test_real": test.values,
            "parametros": "SARIMA(1,1,1)(1,1,1)[7]",
            "aic": round(resultado.aic, 2),
        }

    except Exception as e:
        log.error(f"  → Error SARIMA: {e}")
        return {"status": "error", "razon": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# MODELO 2: PROPHET
# ═══════════════════════════════════════════════════════════════════════════

def entrenar_prophet(df_serie: pd.DataFrame, horizonte: int = HORIZONTE_DIAS) -> dict:
    """
    Entrena Prophet (Facebook) sobre la serie temporal.
    Requiere columnas 'fecha' y 'unidades'.
    """
    log.info("─── Entrenando Prophet ───")
    try:
        from prophet import Prophet
    except ImportError:
        log.warning("  → Prophet no instalado. Intentando con fbprophet...")
        try:
            from fbprophet import Prophet
        except ImportError:
            return {"status": "skip", "razon": "Prophet no disponible"}

    n = len(df_serie)
    if n < MIN_DATOS_SARIMA:
        return {"status": "skip", "razon": "datos insuficientes"}

    # Preparar formato Prophet
    df_p = df_serie[["fecha", "unidades"]].rename(
        columns={"fecha": "ds", "unidades": "y"}
    ).copy()
    df_p["ds"] = pd.to_datetime(df_p["ds"])

    test_size = min(horizonte, max(7, n // 5))
    train_p = df_p.iloc[:-test_size]
    test_p = df_p.iloc[-test_size:]

    try:
        modelo = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,
            interval_width=0.60,
        )
        # Suprimir logs de Prophet
        modelo.fit(train_p)

        # Predicción sobre test
        future_test = modelo.make_future_dataframe(periods=test_size, freq="D")
        pred_full = modelo.predict(future_test)
        pred_test = pred_full.tail(test_size)["yhat"].values
        pred_test = np.maximum(pred_test, 0)

        metricas = calcular_metricas(test_p["y"].values, pred_test)

        # Reentrenar con todo para forecast
        modelo_full = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,
            interval_width=0.60,
        )
        modelo_full.fit(df_p)

        future = modelo_full.make_future_dataframe(periods=horizonte, freq="D")
        forecast_df = modelo_full.predict(future)
        forecast = forecast_df.tail(horizonte)["yhat"].values
        forecast = np.maximum(forecast, 0)

        log.info(f"  → MAE: {metricas['mae']:.2f} | MAPE: {metricas['mape']:.1f}% | R²: {metricas['r2']:.3f}")

        return {
            "status": "ok",
            "nombre": "Prophet",
            "metricas": metricas,
            "forecast": forecast,
            "pred_test": pred_test,
            "test_real": test_p["y"].values,
            "parametros": "weekly_seas=mult, cp_prior=0.05",
        }

    except Exception as e:
        log.error(f"  → Error Prophet: {e}")
        return {"status": "error", "razon": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# MODELO 3: RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════

def entrenar_random_forest(df_features: pd.DataFrame, horizonte: int = HORIZONTE_DIAS) -> dict:
    """
    Entrena Random Forest con los features del ETL.
    """
    log.info("─── Entrenando Random Forest ───")
    from sklearn.ensemble import RandomForestRegressor

    n = len(df_features)
    if n < MIN_DATOS_RF:
        return {"status": "skip", "razon": f"necesita al menos {MIN_DATOS_RF} días"}

    # Seleccionar features
    feature_cols = [
        "dia_semana", "es_fin_semana", "mes",
        "lag_1", "lag_7", "lag_14", "lag_21", "lag_28",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
        "rolling_std_7", "rolling_std_14",
        "rolling_max_7", "rolling_min_7",
        "cambio_semanal_pct", "momentum_7d",
        "hm_indice_semanal", "hm_indice_mensual", "hm_indice_combinado",
        "dia_sin", "dia_cos", "mes_sin", "mes_cos",
        "dias_desde_inicio", "tendencia_norm",
    ]
    # Usar solo las que existen
    feature_cols = [c for c in feature_cols if c in df_features.columns]
    target = "unidades"

    df_ml = df_features[feature_cols + [target, "fecha"]].dropna().copy()
    if len(df_ml) < MIN_DATOS_RF:
        return {"status": "skip", "razon": "datos insuficientes tras dropna"}

    test_size = min(horizonte, max(7, len(df_ml) // 5))
    train_df = df_ml.iloc[:-test_size]
    test_df = df_ml.iloc[-test_size:]

    X_train = train_df[feature_cols].values
    y_train = train_df[target].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target].values

    try:
        modelo = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        modelo.fit(X_train, y_train)

        pred_test = modelo.predict(X_test)
        pred_test = np.maximum(pred_test, 0)

        metricas = calcular_metricas(y_test, pred_test)

        # Feature importance
        importancias = dict(zip(feature_cols, modelo.feature_importances_))
        top_features = sorted(importancias.items(), key=lambda x: x[1], reverse=True)[:5]

        # Forecast futuro: generar features sintéticos
        forecast = _generar_forecast_rf(modelo, df_ml, feature_cols, horizonte)

        log.info(f"  → MAE: {metricas['mae']:.2f} | MAPE: {metricas['mape']:.1f}% | R²: {metricas['r2']:.3f}")
        log.info(f"  → Top features: {[f'{k}: {v:.3f}' for k, v in top_features]}")

        return {
            "status": "ok",
            "nombre": "Random Forest",
            "metricas": metricas,
            "forecast": forecast,
            "pred_test": pred_test,
            "test_real": y_test,
            "parametros": "n_est=200, max_d=6, min_split=5",
            "importancia_features": importancias,
        }

    except Exception as e:
        log.error(f"  → Error Random Forest: {e}")
        return {"status": "error", "razon": str(e)}


def _generar_forecast_rf(
    modelo, df_hist: pd.DataFrame, feature_cols: list, horizonte: int
) -> np.ndarray:
    """Genera forecast iterativo con Random Forest (autoregresivo)."""
    ultimo_dia = pd.to_datetime(df_hist["fecha"].max())
    serie_extendida = df_hist[["fecha", "unidades"]].copy()
    predicciones = []

    for i in range(horizonte):
        nueva_fecha = ultimo_dia + pd.Timedelta(days=i + 1)
        dia_semana = nueva_fecha.dayofweek
        mes = nueva_fecha.month

        # Construir fila de features
        unidades_hist = serie_extendida["unidades"].values
        fila = {}

        fila["dia_semana"] = dia_semana
        fila["es_fin_semana"] = 1 if dia_semana in [5, 6] else 0
        fila["mes"] = mes

        # Lags (usando serie extendida con predicciones previas)
        for lag in [1, 7, 14, 21, 28]:
            idx = len(unidades_hist) - lag
            fila[f"lag_{lag}"] = unidades_hist[idx] if idx >= 0 else 0

        # Rolling stats
        for v in [7, 14, 30]:
            ventana = unidades_hist[-v:] if len(unidades_hist) >= v else unidades_hist
            fila[f"rolling_mean_{v}"] = np.mean(ventana)
            fila[f"rolling_std_{v}"] = np.std(ventana) if len(ventana) > 1 else 0

        fila["rolling_max_7"] = np.max(unidades_hist[-7:]) if len(unidades_hist) >= 7 else np.max(unidades_hist)
        fila["rolling_min_7"] = np.min(unidades_hist[-7:]) if len(unidades_hist) >= 7 else np.min(unidades_hist)

        # Cambio semanal
        if len(unidades_hist) >= 14:
            m1 = np.mean(unidades_hist[-7:])
            m2 = np.mean(unidades_hist[-14:-7])
            fila["cambio_semanal_pct"] = ((m1 - m2) / m2 * 100) if m2 != 0 else 0
        else:
            fila["cambio_semanal_pct"] = 0

        fila["momentum_7d"] = (unidades_hist[-1] - unidades_hist[-7]) if len(unidades_hist) >= 7 else 0

        # H&M
        fila["hm_indice_semanal"] = HM_INDICE_SEMANAL.get(dia_semana, 1.0)
        fila["hm_indice_mensual"] = HM_INDICE_MENSUAL.get(mes, 1.0)
        fila["hm_indice_combinado"] = (
            fila["hm_indice_semanal"]
            * fila["hm_indice_mensual"]
            * DTF_CORRECCION_MENSUAL.get(mes, 1.0)
        )

        # Cíclicos
        fila["dia_sin"] = np.sin(2 * np.pi * dia_semana / 7)
        fila["dia_cos"] = np.cos(2 * np.pi * dia_semana / 7)
        fila["mes_sin"] = np.sin(2 * np.pi * mes / 12)
        fila["mes_cos"] = np.cos(2 * np.pi * mes / 12)

        dias_total = (nueva_fecha - pd.to_datetime(df_hist["fecha"].min())).days
        fila["dias_desde_inicio"] = dias_total
        max_d = df_hist["dias_desde_inicio"].max() if "dias_desde_inicio" in df_hist.columns else dias_total
        fila["tendencia_norm"] = dias_total / max_d if max_d > 0 else 0

        # Predecir
        X_new = np.array([[fila.get(c, 0) for c in feature_cols]])
        pred = max(0, modelo.predict(X_new)[0])
        predicciones.append(pred)

        # Agregar a serie extendida
        nueva_fila = pd.DataFrame({"fecha": [nueva_fecha], "unidades": [pred]})
        serie_extendida = pd.concat([serie_extendida, nueva_fila], ignore_index=True)

    return np.array(predicciones)


# ═══════════════════════════════════════════════════════════════════════════
# MODELO 4: TRANSFERENCIA H&M
# ═══════════════════════════════════════════════════════════════════════════

def generar_forecast_hm(
    df_serie: pd.DataFrame, horizonte: int = HORIZONTE_DIAS
) -> dict:
    """
    Genera forecast usando transferencia de patrones estacionales de H&M,
    calibrados a la escala real de DTF Fashion.
    """
    log.info("─── Generando Forecast por Transferencia H&M ───")

    serie = df_serie[["fecha", "unidades"]].copy()
    dias_activos = serie[serie["unidades"] > 0]

    if len(dias_activos) == 0:
        return {"status": "skip", "razon": "sin ventas registradas"}

    promedio_diario = dias_activos["unidades"].mean()
    ultima_fecha = pd.to_datetime(serie["fecha"].max())

    # Calcular correcciones propias por mes (si hay datos)
    correcciones = DTF_CORRECCION_MENSUAL.copy()
    serie["mes"] = pd.to_datetime(serie["fecha"]).dt.month
    for mes, grupo in serie.groupby("mes"):
        real_mes = grupo["unidades"].mean()
        if real_mes > 0:
            estimado = HM_INDICE_MENSUAL.get(mes, 1.0) * promedio_diario
            if estimado > 0:
                correcciones[mes] = real_mes / estimado

    # Generar forecast
    fechas_futuras = []
    predicciones = []
    for i in range(horizonte):
        fecha = ultima_fecha + pd.Timedelta(days=i + 1)
        dia_sem = fecha.dayofweek
        mes = fecha.month

        idx_sem = HM_INDICE_SEMANAL.get(dia_sem, 1.0)
        idx_mes = HM_INDICE_MENSUAL.get(mes, 1.0)
        corr = correcciones.get(mes, 1.0)

        pred = promedio_diario * idx_sem * idx_mes * corr
        pred = max(0, pred)

        fechas_futuras.append(fecha)
        predicciones.append(pred)

    forecast = np.array(predicciones)

    # Evaluar sobre datos históricos (backtest)
    if len(serie) >= horizonte:
        backtest_real = serie["unidades"].values[-horizonte:]
        backtest_pred = []
        for i, row in serie.tail(horizonte).iterrows():
            f = pd.to_datetime(row["fecha"])
            ds = f.dayofweek
            m = f.month
            bp = promedio_diario * HM_INDICE_SEMANAL.get(ds, 1.0) * HM_INDICE_MENSUAL.get(m, 1.0) * correcciones.get(m, 1.0)
            backtest_pred.append(max(0, bp))
        metricas = calcular_metricas(backtest_real, np.array(backtest_pred))
    else:
        metricas = {"mae": 0, "mape": 0, "r2": 0}

    log.info(f"  → Promedio diario DTF: {promedio_diario:.2f} uds")
    log.info(f"  → Forecast 30d total: {forecast.sum():.0f} uds (central)")
    log.info(f"  → MAE backtest: {metricas['mae']:.2f} | MAPE: {metricas['mape']:.1f}%")

    return {
        "status": "ok",
        "nombre": "Transferencia H&M",
        "metricas": metricas,
        "forecast": forecast,
        "fechas": fechas_futuras,
        "parametros": f"prom_diario={promedio_diario:.2f}, corr_propia=True",
        "promedio_diario": promedio_diario,
        "correcciones": correcciones,
    }


# ═══════════════════════════════════════════════════════════════════════════
# COMPARACIÓN Y SELECCIÓN
# ═══════════════════════════════════════════════════════════════════════════

def comparar_modelos(resultados: list, baseline: dict) -> dict:
    """
    Compara modelos entrenados, selecciona ganador, calcula mejora vs baseline.
    """
    log.info("═══ COMPARACIÓN DE MODELOS ═══")

    modelos_ok = [r for r in resultados if r.get("status") == "ok"]
    if not modelos_ok:
        log.error("  → Ningún modelo se entrenó exitosamente")
        return {"ganador": None, "comparacion": [], "baseline": baseline}

    comparacion = []
    for m in modelos_ok:
        met = m["metricas"]
        mejora_mae = ((baseline["mae"] - met["mae"]) / baseline["mae"] * 100) if baseline["mae"] > 0 else 0

        info = {
            "nombre": m["nombre"],
            "mae": met["mae"],
            "mape": met["mape"],
            "r2": met["r2"],
            "mejora_vs_baseline_pct": round(mejora_mae, 1),
            "cumple_objetivo": mejora_mae >= 20,
        }
        comparacion.append(info)

        log.info(
            f"  {m['nombre']:20s} | MAE: {met['mae']:8.2f} | "
            f"MAPE: {met['mape']:6.1f}% | R²: {met['r2']:6.3f} | "
            f"Mejora: {mejora_mae:+.1f}%"
        )

    # Seleccionar ganador por menor MAPE (métrica principal del proyecto)
    comparacion.sort(key=lambda x: x["mape"])
    ganador = comparacion[0]

    log.info(f"\n  🏆 GANADOR: {ganador['nombre']} (MAPE: {ganador['mape']:.1f}%)")
    log.info(f"  📊 Baseline naive MAE: {baseline['mae']:.2f} | MAPE: {baseline['mape']:.1f}%")
    log.info(f"  📈 Mejora vs baseline: {ganador['mejora_vs_baseline_pct']:+.1f}%")

    return {
        "ganador": ganador["nombre"],
        "comparacion": comparacion,
        "baseline": baseline,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ESCRITURA A POSTGRESQL
# ═══════════════════════════════════════════════════════════════════════════

def guardar_resultados(
    resultados: list,
    comparacion: dict,
    df_serie: pd.DataFrame,
) -> str:
    """Guarda predicciones, métricas y training run a PostgreSQL."""
    log.info("Guardando resultados a base de datos...")
    db = engine()
    run_id = str(uuid.uuid4())[:8]
    ahora = datetime.now()
    ultima_fecha = pd.to_datetime(df_serie["fecha"].max())

    # ── Training Run ──
    run_data = pd.DataFrame([{
        "run_id": run_id,
        "fecha_ejecucion": ahora,
        "modelo_ganador": comparacion.get("ganador", "N/A"),
        "n_datos": len(df_serie),
        "horizonte": HORIZONTE_DIAS,
        "baseline_mae": comparacion["baseline"]["mae"],
        "baseline_mape": comparacion["baseline"]["mape"],
        "mejor_mae": comparacion["comparacion"][0]["mae"] if comparacion["comparacion"] else None,
        "mejor_mape": comparacion["comparacion"][0]["mape"] if comparacion["comparacion"] else None,
        "mejora_pct": comparacion["comparacion"][0]["mejora_vs_baseline_pct"] if comparacion["comparacion"] else None,
        "created_at": ahora,
    }])
    write_dataframe(run_data, "training_runs", if_exists="append")
    log.info(f"  → training_runs: run_id={run_id}")

    # ── Métricas por modelo ──
    metricas_rows = []
    for m in [r for r in resultados if r.get("status") == "ok"]:
        met = m["metricas"]
        metricas_rows.append({
            "run_id": run_id,
            "modelo": m["nombre"],
            "mae": met["mae"],
            "mape": met["mape"],
            "r2": met["r2"],
            "parametros": m.get("parametros", ""),
            "es_ganador": m["nombre"] == comparacion.get("ganador"),
            "created_at": ahora,
        })
    if metricas_rows:
        write_dataframe(pd.DataFrame(metricas_rows), "metricas_modelos", if_exists="append")
        log.info(f"  → metricas_modelos: {len(metricas_rows)} modelos")

    # ── Predicciones del modelo ganador ──
    ganador_data = next(
        (r for r in resultados if r.get("nombre") == comparacion.get("ganador")),
        None,
    )
    if ganador_data and "forecast" in ganador_data:
        forecast = ganador_data["forecast"]
        fechas = [ultima_fecha + pd.Timedelta(days=i + 1) for i in range(len(forecast))]

        pred_rows = []
        for i, (fecha, pred) in enumerate(zip(fechas, forecast)):
            pred_rows.append({
                "run_id": run_id,
                "modelo": ganador_data["nombre"],
                "fecha_prediccion": fecha,
                "unidades_predichas": round(float(pred), 2),
                "banda_inferior": round(float(pred * (1 - BANDA_INCERTIDUMBRE)), 2),
                "banda_superior": round(float(pred * (1 + BANDA_INCERTIDUMBRE)), 2),
                "dia_horizonte": i + 1,
                "created_at": ahora,
            })

        # También guardar predicciones de todos los modelos OK
        for m in [r for r in resultados if r.get("status") == "ok" and r["nombre"] != ganador_data["nombre"]]:
            fc = m["forecast"]
            for i, (fecha, pred) in enumerate(zip(fechas[:len(fc)], fc)):
                pred_rows.append({
                    "run_id": run_id,
                    "modelo": m["nombre"],
                    "fecha_prediccion": fecha,
                    "unidades_predichas": round(float(pred), 2),
                    "banda_inferior": round(float(pred * (1 - BANDA_INCERTIDUMBRE)), 2),
                    "banda_superior": round(float(pred * (1 + BANDA_INCERTIDUMBRE)), 2),
                    "dia_horizonte": i + 1,
                    "created_at": ahora,
                })

        # Limpiar predicciones anteriores
        with db.begin() as conn:
            conn.execute(text("DELETE FROM predicciones"))
        write_dataframe(pd.DataFrame(pred_rows), "predicciones", if_exists="append")
        log.info(f"  → predicciones: {len(pred_rows)} registros")

    return run_id


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL DE ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════════

def ejecutar_entrenamiento() -> dict:
    """
    Pipeline completo:
      1. Lee datos de PostgreSQL (serie_semanal + features)
      2. Entrena SARIMA, Prophet, Random Forest
      3. Calcula baseline naive
      4. Compara los 3 modelos
      5. Guarda resultados a PostgreSQL

    Los índices estacionales de retail se usan internamente como
    features del Random Forest, no como modelo independiente.

    Returns:
        dict con resumen del entrenamiento
    """
    log.info("=" * 70)
    log.info("INICIO ENTRENAMIENTO v4.0 — DTF Fashion")
    log.info("=" * 70)
    t0 = datetime.now()

    # ── 1. Leer datos de PostgreSQL ──
    log.info("Leyendo datos de PostgreSQL...")
    try:
        df_serie = read_sql("SELECT * FROM serie_semanal ORDER BY fecha")
        df_features = read_sql("SELECT * FROM features ORDER BY fecha")
    except Exception as e:
        log.error(f"Error leyendo datos: {e}")
        return {"status": "error", "mensaje": "No hay datos. Ejecuta primero el ETL."}

    if df_serie.empty:
        return {"status": "error", "mensaje": "Tabla serie_semanal vacía. Sube datos primero."}

    df_serie["fecha"] = pd.to_datetime(df_serie["fecha"])
    df_features["fecha"] = pd.to_datetime(df_features["fecha"])
    serie = df_serie.set_index("fecha")["unidades"]

    log.info(f"  → {len(df_serie)} días de datos")
    log.info(f"  → {serie.sum():.0f} unidades totales")

    # ── 1b. Fine-tuning H&M — recalcular índices con datos actualizados ──
    # Actualiza los globals del módulo para que TODOS los modelos usen
    # los índices frescos en este ciclo de entrenamiento.
    global HM_INDICE_SEMANAL, HM_INDICE_MENSUAL, DTF_CORRECCION_MENSUAL
    try:
        nuevos_indices = calcular_indices_hm(df_serie)
        guardar_indices(nuevos_indices)
        HM_INDICE_SEMANAL      = nuevos_indices["indice_semanal"]
        HM_INDICE_MENSUAL      = nuevos_indices["indice_mensual"]
        DTF_CORRECCION_MENSUAL = nuevos_indices["correccion_mensual"]
        log.info(
            "Fine-tuning H&M completado — origen: %s | obs_dtf: %d | "
            "factor_escala: %.8f",
            nuevos_indices["origen"],
            nuevos_indices["n_observaciones_dtf"],
            nuevos_indices["factor_escala"],
        )
    except Exception as exc:
        log.warning(
            "Fine-tuning H&M falló (%s) — continuando con índices previos", exc
        )

    # ── 2. Baseline naive ──
    baseline = calcular_baseline_naive(serie, HORIZONTE_DIAS)
    log.info(f"\n📌 Baseline naive — MAE: {baseline['mae']:.2f} | MAPE: {baseline['mape']:.1f}%\n")

    # ── 3. Entrenar modelos ──
    resultados = []

    # SARIMA
    res_sarima = entrenar_sarima(serie, HORIZONTE_DIAS)
    resultados.append(res_sarima)

    # Prophet
    res_prophet = entrenar_prophet(df_serie, HORIZONTE_DIAS)
    resultados.append(res_prophet)

    # Random Forest (usa índices estacionales de retail como features internos)
    res_rf = entrenar_random_forest(df_features, HORIZONTE_DIAS)
    resultados.append(res_rf)

    # ── 4. Comparar ──
    comparacion = comparar_modelos(resultados, baseline)

    # ── 5. Guardar a DB ──
    run_id = guardar_resultados(resultados, comparacion, df_serie)

    elapsed = (datetime.now() - t0).total_seconds()

    resumen = {
        "status": "ok",
        "run_id": run_id,
        "modelos_entrenados": [r["nombre"] for r in resultados if r.get("status") == "ok"],
        "modelos_fallidos": [
            {"nombre": r.get("nombre", "?"), "razon": r.get("razon", "")}
            for r in resultados if r.get("status") != "ok"
        ],
        "ganador": comparacion.get("ganador"),
        "comparacion": comparacion.get("comparacion", []),
        "baseline": baseline,
        "horizonte_dias": HORIZONTE_DIAS,
        "banda_incertidumbre": BANDA_INCERTIDUMBRE,
        "tiempo_seg": round(elapsed, 2),
    }

    log.info("=" * 70)
    log.info(f"ENTRENAMIENTO COMPLETADO en {elapsed:.1f}s")
    log.info(f"  Run ID: {run_id}")
    log.info(f"  Ganador: {comparacion.get('ganador')}")
    log.info("=" * 70)

    return resumen


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    resultado = ejecutar_entrenamiento()
    print("\n✅ Resultado:")
    for k, v in resultado.items():
        if k != "comparacion":
            print(f"  {k}: {v}")
    print("\n📊 Comparación de modelos:")
    for m in resultado.get("comparacion", []):
        flag = "🏆" if m["nombre"] == resultado.get("ganador") else "  "
        print(f"  {flag} {m['nombre']:20s} | MAE: {m['mae']:.2f} | MAPE: {m['mape']:.1f}% | Mejora: {m['mejora_vs_baseline_pct']:+.1f}%")