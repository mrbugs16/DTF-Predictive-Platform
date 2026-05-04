"""
ETL Pipeline v4.0 — DTF Fashion Predictive Analytics Platform
Acepta Excel/CSV del usuario, limpia, genera features, escribe a PostgreSQL.
"""

import os
import sys
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

# ── Agregar raíz del proyecto al path ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.connection import engine, read_sql, write_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("etl_pipeline")

# ═══════════════════════════════════════════════════════════════════════════
# ÍNDICES ESTACIONALES H&M HARDCODEADOS
# Extraídos de 31.7M transacciones — SARIMA(1,1,1)(1,1,1)[7], MAPE 15.48%
# ═══════════════════════════════════════════════════════════════════════════
HM_INDICE_SEMANAL = {
    0: 0.94,   # Lunes
    1: 0.89,   # Martes
    2: 0.91,   # Miércoles
    3: 0.95,   # Jueves
    4: 1.05,   # Viernes
    5: 1.16,   # Sábado  (pico H&M: 48,383 uds promedio)
    6: 1.10,   # Domingo
}

HM_INDICE_MENSUAL = {
    1:  0.92,   # Enero
    2:  0.90,   # Febrero
    3:  0.86,   # Marzo
    4:  0.95,   # Abril
    5:  1.14,   # Mayo
    6:  1.41,   # Junio   (pico H&M: 61,000 uds)
    7:  1.16,   # Julio
    8:  0.95,   # Agosto
    9:  1.03,   # Septiembre
    10: 0.93,   # Octubre
    11: 0.93,   # Noviembre
    12: 0.84,   # Diciembre (mínimo H&M: 36,500 uds)
}

# Factores de corrección DTF México vs H&M Europa
# Calculados en research/dtf_finetuning.py
DTF_CORRECCION_MENSUAL = {
    1:  1.100,  # Enero   — ajuste moderado
    2:  1.529,  # Febrero — 53% más que patrón H&M
    3:  1.754,  # Marzo   — 75% más (temporada fuerte México)
    4:  1.200,  # Abril   — estimado (sin datos suficientes)
    5:  1.000,  # Mayo    — sin corrección (sin datos)
    6:  1.000,  # Junio
    7:  1.000,  # Julio
    8:  1.000,  # Agosto
    9:  1.000,  # Septiembre
    10: 0.704,  # Octubre  — arranque negocio, bajo vs H&M
    11: 0.702,  # Noviembre
    12: 0.900,  # Diciembre — estimado
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════

def cargar_archivo(ruta: str) -> pd.DataFrame:
    """Carga Excel (.xlsx/.xls) o CSV y normaliza columnas."""
    ruta = Path(ruta)
    log.info(f"Cargando archivo: {ruta.name}")

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró: {ruta}")

    ext = ruta.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(ruta)
        # Detectar si los headers están mal (todas las columnas son "Unnamed")
        if all("Unnamed" in str(c) or "unnamed" in str(c) for c in df.columns):
            log.warning("  → Headers no detectados en fila 0, buscando fila de headers...")
            for skip in range(1, 6):
                df_test = pd.read_excel(ruta, skiprows=skip)
                unnamed_count = sum(1 for c in df_test.columns if "Unnamed" in str(c) or "unnamed" in str(c))
                if unnamed_count < len(df_test.columns) * 0.5:
                    df = df_test
                    log.info(f"  → Headers encontrados en fila {skip + 1}")
                    break
        # Eliminar columnas completamente vacías
        df = df.dropna(axis=1, how="all")
    elif ext == ".csv":
        # Intentar detectar separador
        for sep in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(ruta, sep=sep, encoding="utf-8")
                if len(df.columns) > 1:
                    break
            except Exception:
                continue
        else:
            df = pd.read_csv(ruta)
    else:
        raise ValueError(f"Formato no soportado: {ext}. Usa .xlsx, .xls o .csv")

    # Normalizar nombres de columnas
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[áà]", "a", regex=True)
        .str.replace(r"[éè]", "e", regex=True)
        .str.replace(r"[íì]", "i", regex=True)
        .str.replace(r"[óò]", "o", regex=True)
        .str.replace(r"[úù]", "u", regex=True)
        .str.replace(r"[ñ]", "n", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )

    log.info(f"  → {len(df)} filas, {len(df.columns)} columnas: {list(df.columns)}")
    return df


def detectar_columnas(df: pd.DataFrame) -> dict:
    """
    Detecta automáticamente qué columnas corresponden a fecha, cantidad, precio, etc.
    Soporta formato Shopify (Date, Quantity, Total, Product, Category)
    y formato libre en español (fecha, cantidad, precio, producto, categoria).
    """
    mapeo = {}

    # ── Fecha ──
    fecha_keywords = ["date", "fecha", "t_dat", "created_at", "order_date", "fecha_de_venta"]
    excluir_fecha = {"dia_semana", "dia_nombre", "dias_desde_inicio", "update_date"}
    # Match exacto primero
    for col in df.columns:
        if col in fecha_keywords:
            mapeo["fecha"] = col
            break
    # Substring match
    if "fecha" not in mapeo:
        for col in df.columns:
            if col in excluir_fecha:
                continue
            if any(k in col for k in ["fecha", "date", "t_dat"]):
                mapeo["fecha"] = col
                break
    # Último recurso: columna con tipo datetime
    if "fecha" not in mapeo:
        for col in df.columns:
            if col in excluir_fecha:
                continue
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > len(df) * 0.5:
                    mapeo["fecha"] = col
                    break
            except Exception:
                continue

    # ── Cantidad ──
    for col in df.columns:
        if any(k == col or k in col for k in ["quantity", "cantidad", "unidades", "qty", "units", "lineitem_quantity"]):
            mapeo["cantidad"] = col
            break

    # ── Precio / Total ──
    # Priorizar "total" sobre "subtotal" sobre "ingreso/precio"
    precio_prioridad = [
        ["total", "total_neto"],
        ["subtotal", "precio", "price", "ingreso", "revenue", "monto", "lineitem_price"],
    ]
    for grupo in precio_prioridad:
        if "precio" in mapeo:
            break
        for col in df.columns:
            if any(k == col or k in col for k in grupo):
                mapeo["precio"] = col
                break

    # ── Producto / diseño ──
    for col in df.columns:
        if any(k == col or k in col for k in ["product", "producto", "diseno", "design", "sku", "article", "nombre", "lineitem_name"]):
            mapeo["producto"] = col
            break

    # ── Categoría ──
    for col in df.columns:
        if any(k == col or k in col for k in ["category", "categoria", "linea", "product_type"]):
            mapeo["categoria"] = col
            break

    log.info(f"  → Mapeo detectado: {mapeo}")
    return mapeo


# ═══════════════════════════════════════════════════════════════════════════
# 2. LIMPIEZA
# ═══════════════════════════════════════════════════════════════════════════

_MESES_ES = {
    "enero": "January",   "febrero": "February", "marzo": "March",
    "abril": "April",     "mayo": "May",          "junio": "June",
    "julio": "July",      "agosto": "August",     "septiembre": "September",
    "octubre": "October", "noviembre": "November","diciembre": "December",
}

def _parsear_fechas_espanol(serie: pd.Series) -> pd.Series:
    """
    Traduce meses en español a inglés antes de parsear.
    Maneja formatos como '07 de octubre 2025' o '31 de octubre de 2025'.
    """
    s = serie.astype(str).str.lower().str.strip()
    for es, en in _MESES_ES.items():
        s = s.str.replace(es, en, case=False, regex=False)
    # Eliminar la(s) partícula(s) "de" que quedan sueltas: "07 de October de 2025"
    s = s.str.replace(r"\bde\b", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def limpiar_datos(df: pd.DataFrame, mapeo: dict) -> pd.DataFrame:
    """Limpia y valida los datos crudos."""
    log.info("Limpiando datos...")
    n_original = len(df)

    # Parsear fecha
    col_fecha = mapeo.get("fecha")
    if col_fecha is None:
        raise ValueError("No se detectó columna de fecha. Asegúrate de tener una columna 'fecha' o 'date'.")

    # Intentar parseo directo
    df["fecha"] = pd.to_datetime(df[col_fecha], errors="coerce", dayfirst=True)

    # Si falló, intentar con traducción de meses en español
    # (ej: "07 de octubre 2025" → "07 October 2025")
    nulos = df["fecha"].isna().sum()
    if nulos > len(df) * 0.3:
        parsed_es = _parsear_fechas_espanol(df[col_fecha])
        if parsed_es.notna().sum() > df["fecha"].notna().sum():
            df["fecha"] = parsed_es
            log.info(f"  → {parsed_es.notna().sum()} fechas parseadas con meses en español")
            nulos = df["fecha"].isna().sum()

    # Si falló (muchos NaT), intentar como serial date de Excel (ej: 45930 = 2025-10-06)
    if nulos > len(df) * 0.5:
        log.warning(f"  → {nulos} fechas no parseadas, intentando como serial Excel...")
        vals = pd.to_numeric(df[col_fecha], errors="coerce")
        if vals.notna().sum() > len(df) * 0.5 and vals.median() > 40000:
            df["fecha"] = pd.to_datetime(vals, unit="D", origin="1899-12-30", errors="coerce")
            log.info(f"  → Convertidas {df['fecha'].notna().sum()} fechas desde serial Excel")

    # Si las fechas son epoch/sospechosas (1970), el mapeo detectó la columna equivocada
    if df["fecha"].notna().any() and df["fecha"].min().year < 2000:
        log.warning(f"  → Fechas sospechosas (año {df['fecha'].min().year}), buscando otra columna...")
        for col in df.columns:
            if col == col_fecha:
                continue
            try:
                test = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
                if test.notna().sum() > len(df) * 0.5 and test.min().year >= 2020:
                    df["fecha"] = test
                    log.info(f"  → Columna de fecha corregida a: '{col}'")
                    break
            except Exception:
                continue

    nulos_fecha = df["fecha"].isna().sum()
    if nulos_fecha > 0:
        log.warning(f"  → {nulos_fecha} filas con fecha inválida eliminadas")
        df = df.dropna(subset=["fecha"])

    # Cantidad
    col_cant = mapeo.get("cantidad")
    if col_cant:
        df["cantidad"] = pd.to_numeric(df[col_cant], errors="coerce").fillna(1).astype(int)
        df = df[df["cantidad"] > 0]
    else:
        df["cantidad"] = 1
        log.warning("  → No se detectó columna de cantidad; asumiendo 1 unidad por fila")

    # Precio
    col_precio = mapeo.get("precio")
    if col_precio:
        df["precio_unitario"] = pd.to_numeric(df[col_precio], errors="coerce").fillna(0)
    else:
        df["precio_unitario"] = 0.0
        log.warning("  → No se detectó columna de precio")

    # Producto
    col_prod = mapeo.get("producto")
    if col_prod:
        df["producto"] = df[col_prod].astype(str).str.strip()
    else:
        df["producto"] = "general"

    # Categoría
    col_cat = mapeo.get("categoria")
    if col_cat:
        df["categoria"] = df[col_cat].astype(str).str.strip().str.title()
    else:
        df["categoria"] = "General"

    # Eliminar duplicados exactos
    n_antes = len(df)
    df = df.drop_duplicates()
    if len(df) < n_antes:
        log.info(f"  → {n_antes - len(df)} duplicados eliminados")

    # Calcular ingreso
    df["ingreso_bruto"] = df["cantidad"] * df["precio_unitario"]

    # Generar ID de venta único
    df["venta_id"] = df.apply(
        lambda r: hashlib.md5(
            f"{r['fecha']}_{r['producto']}_{r['cantidad']}_{r.name}".encode()
        ).hexdigest()[:12],
        axis=1,
    )

    # Ordenar por fecha
    df = df.sort_values("fecha").reset_index(drop=True)

    log.info(f"  → Limpieza completa: {n_original} → {len(df)} filas")
    log.info(f"  → Período: {df['fecha'].min().date()} a {df['fecha'].max().date()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. AGREGACIÓN A SERIE TEMPORAL
# ═══════════════════════════════════════════════════════════════════════════

def agregar_serie_semanal(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ventas diarias y rellena días sin actividad."""
    log.info("Generando serie temporal diaria...")

    _fmin = df["fecha"].min()
    _fmax = df["fecha"].max()
    if pd.isna(_fmin) or pd.isna(_fmax):
        raise ValueError(
            "No se encontraron fechas válidas tras la limpieza. "
            "Verifica que el archivo tenga una columna 'fecha' con fechas legibles."
        )
    fecha_min = _fmin.normalize()
    fecha_max = _fmax.normalize()

    # Crear rango completo de fechas
    rango = pd.date_range(start=fecha_min, end=fecha_max, freq="D")
    serie = pd.DataFrame({"fecha": rango})

    # Agregar por día
    diario = (
        df.groupby(df["fecha"].dt.normalize())
        .agg(
            unidades=("cantidad", "sum"),
            ingreso_bruto=("ingreso_bruto", "sum"),
            num_transacciones=("venta_id", "nunique"),
            productos_unicos=("producto", "nunique"),
        )
        .reset_index()
        .rename(columns={"fecha": "fecha"})
    )

    serie = serie.merge(diario, on="fecha", how="left")
    serie["unidades"] = serie["unidades"].fillna(0).astype(int)
    serie["ingreso_bruto"] = serie["ingreso_bruto"].fillna(0.0)
    serie["num_transacciones"] = serie["num_transacciones"].fillna(0).astype(int)
    serie["productos_unicos"] = serie["productos_unicos"].fillna(0).astype(int)

    # Campos temporales
    serie["dia_semana"] = serie["fecha"].dt.dayofweek
    serie["dia_nombre"] = serie["fecha"].dt.day_name()
    serie["semana_iso"] = serie["fecha"].dt.isocalendar().week.astype(int)
    serie["mes"] = serie["fecha"].dt.month
    serie["anio"] = serie["fecha"].dt.year
    serie["es_fin_semana"] = serie["dia_semana"].isin([5, 6]).astype(int)

    # Ingreso acumulado
    serie["ingreso_acumulado"] = serie["ingreso_bruto"].cumsum()

    log.info(f"  → Serie de {len(serie)} días ({serie['unidades'].sum()} unidades totales)")
    log.info(f"  → Días con actividad: {(serie['unidades'] > 0).sum()} de {len(serie)}")
    return serie


# ═══════════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def generar_features(serie: pd.DataFrame) -> pd.DataFrame:
    """Genera features para los modelos ML incluyendo índices H&M."""
    log.info("Generando features...")
    f = serie[["fecha", "unidades", "dia_semana", "mes", "es_fin_semana"]].copy()

    # ── Lags ──
    for lag in [1, 7, 14, 21, 28]:
        f[f"lag_{lag}"] = f["unidades"].shift(lag)

    # ── Rolling stats ──
    for ventana in [7, 14, 30]:
        f[f"rolling_mean_{ventana}"] = (
            f["unidades"].rolling(window=ventana, min_periods=1).mean()
        )
        f[f"rolling_std_{ventana}"] = (
            f["unidades"].rolling(window=ventana, min_periods=1).std().fillna(0)
        )

    # ── Rolling max y min (ventana 7) ──
    f["rolling_max_7"] = f["unidades"].rolling(window=7, min_periods=1).max()
    f["rolling_min_7"] = f["unidades"].rolling(window=7, min_periods=1).min()

    # ── Cambio semanal (%) ──
    media_actual = f["unidades"].rolling(window=7, min_periods=1).mean()
    media_anterior = f["unidades"].shift(7).rolling(window=7, min_periods=1).mean()
    f["cambio_semanal_pct"] = ((media_actual - media_anterior) / media_anterior.replace(0, np.nan) * 100).fillna(0)

    # ── Momentum (diferencia absoluta vs semana pasada) ──
    f["momentum_7d"] = f["unidades"] - f["unidades"].shift(7)
    f["momentum_7d"] = f["momentum_7d"].fillna(0)

    # ── Índices estacionales H&M ──
    f["hm_indice_semanal"] = f["dia_semana"].map(HM_INDICE_SEMANAL)
    f["hm_indice_mensual"] = f["mes"].map(HM_INDICE_MENSUAL)
    f["hm_correccion_dtf"] = f["mes"].map(DTF_CORRECCION_MENSUAL)

    # Índice combinado H&M calibrado
    f["hm_indice_combinado"] = (
        f["hm_indice_semanal"] * f["hm_indice_mensual"] * f["hm_correccion_dtf"]
    )

    # ── Encoding cíclico para día y mes ──
    f["dia_sin"] = np.sin(2 * np.pi * f["dia_semana"] / 7)
    f["dia_cos"] = np.cos(2 * np.pi * f["dia_semana"] / 7)
    f["mes_sin"] = np.sin(2 * np.pi * f["mes"] / 12)
    f["mes_cos"] = np.cos(2 * np.pi * f["mes"] / 12)

    # ── Días desde inicio ──
    f["dias_desde_inicio"] = (f["fecha"] - f["fecha"].min()).dt.days

    # ── Tendencia lineal normalizada [0,1] ──
    max_dias = f["dias_desde_inicio"].max()
    f["tendencia_norm"] = f["dias_desde_inicio"] / max_dias if max_dias > 0 else 0

    # ── Rellenar NaN restantes ──
    f = f.fillna(0)

    n_features = len([c for c in f.columns if c not in ("fecha", "unidades")])
    log.info(f"  → {n_features} features generados")
    return f


# ═══════════════════════════════════════════════════════════════════════════
# 5. ESCRITURA A POSTGRESQL
# ═══════════════════════════════════════════════════════════════════════════

def escribir_a_db(
    df_ventas: pd.DataFrame,
    df_serie: pd.DataFrame,
    df_features: pd.DataFrame,
) -> dict:
    """Escribe los DataFrames procesados a PostgreSQL."""
    log.info("Escribiendo a base de datos...")
    db = engine()
    resumen = {}

    # ── Tabla: ventas ──
    cols_ventas = [
        "venta_id", "fecha", "producto", "categoria",
        "cantidad", "precio_unitario", "ingreso_bruto",
    ]
    df_v = df_ventas[[c for c in cols_ventas if c in df_ventas.columns]].copy()
    df_v["created_at"] = datetime.now()

    # Limpiar tabla existente e insertar
    with db.begin() as conn:
        conn.execute(text("DELETE FROM ventas"))
    write_dataframe(df_v, "ventas", if_exists="append")
    resumen["ventas"] = len(df_v)
    log.info(f"  → ventas: {len(df_v)} registros")

    # ── Tabla: serie_semanal ──
    cols_serie = [
        "fecha", "unidades", "ingreso_bruto", "num_transacciones",
        "productos_unicos", "dia_semana", "dia_nombre", "semana_iso",
        "mes", "anio", "es_fin_semana", "ingreso_acumulado",
    ]
    df_s = df_serie[[c for c in cols_serie if c in df_serie.columns]].copy()
    df_s["created_at"] = datetime.now()

    with db.begin() as conn:
        conn.execute(text("DELETE FROM serie_semanal"))
    write_dataframe(df_s, "serie_semanal", if_exists="append")
    resumen["serie_semanal"] = len(df_s)
    log.info(f"  → serie_semanal: {len(df_s)} registros")

    # ── Tabla: features ──
    df_f = df_features.copy()
    df_f["created_at"] = datetime.now()

    with db.begin() as conn:
        conn.execute(text("DELETE FROM features"))
    write_dataframe(df_f, "features", if_exists="append")
    resumen["features"] = len(df_f)
    log.info(f"  → features: {len(df_f)} registros")

    # ── Tabla: factores_hm ──
    factores = []
    for dia, idx in HM_INDICE_SEMANAL.items():
        factores.append({
            "tipo": "semanal",
            "clave": str(dia),
            "valor": idx,
            "descripcion": f"Índice día {dia} (0=Lun, 6=Dom)",
        })
    for mes, idx in HM_INDICE_MENSUAL.items():
        factores.append({
            "tipo": "mensual",
            "clave": str(mes),
            "valor": idx,
            "descripcion": f"Índice mes {mes}",
        })
    for mes, corr in DTF_CORRECCION_MENSUAL.items():
        factores.append({
            "tipo": "correccion_dtf",
            "clave": str(mes),
            "valor": corr,
            "descripcion": f"Corrección DTF mes {mes}",
        })

    df_hm = pd.DataFrame(factores)
    df_hm["created_at"] = datetime.now()

    with db.begin() as conn:
        conn.execute(text("DELETE FROM factores_hm"))
    write_dataframe(df_hm, "factores_hm", if_exists="append")
    resumen["factores_hm"] = len(df_hm)
    log.info(f"  → factores_hm: {len(df_hm)} registros")

    return resumen


# ═══════════════════════════════════════════════════════════════════════════
# 6. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def ejecutar_pipeline(ruta_archivo: str) -> dict:
    """
    Pipeline ETL completo:
      1. Carga archivo Excel/CSV
      2. Detecta columnas automáticamente
      3. Limpia y valida
      4. Agrega serie temporal diaria
      5. Genera features (lags, rolling, H&M)
      6. Escribe todo a PostgreSQL

    Returns:
        dict con resumen de registros insertados y metadata
    """
    log.info("=" * 70)
    log.info("INICIO PIPELINE ETL v4.0 — DTF Fashion")
    log.info("=" * 70)
    t0 = datetime.now()

    # Paso 1: Cargar
    df_raw = cargar_archivo(ruta_archivo)

    # Paso 2: Detectar columnas
    mapeo = detectar_columnas(df_raw)

    # Paso 3: Limpiar
    df_limpio = limpiar_datos(df_raw, mapeo)

    # Paso 4: Agregar serie diaria
    df_serie = agregar_serie_semanal(df_limpio)

    # Paso 5: Features
    df_features = generar_features(df_serie)

    # Paso 6: Escribir a DB
    resumen_db = escribir_a_db(df_limpio, df_serie, df_features)

    elapsed = (datetime.now() - t0).total_seconds()

    resultado = {
        "status": "ok",
        "archivo": str(ruta_archivo),
        "filas_crudas": len(df_raw),
        "filas_limpias": len(df_limpio),
        "dias_serie": len(df_serie),
        "features_generados": len([
            c for c in df_features.columns if c not in ("fecha", "unidades")
        ]),
        "periodo": {
            "inicio": str(df_limpio["fecha"].min().date()),
            "fin": str(df_limpio["fecha"].max().date()),
        },
        "unidades_totales": int(df_limpio["cantidad"].sum()),
        "ingreso_total": float(df_limpio["ingreso_bruto"].sum()),
        "registros_db": resumen_db,
        "tiempo_seg": round(elapsed, 2),
    }

    log.info("=" * 70)
    log.info(f"PIPELINE COMPLETADO en {elapsed:.1f}s")
    log.info(f"  Filas: {resultado['filas_crudas']} → {resultado['filas_limpias']}")
    log.info(f"  Serie: {resultado['dias_serie']} días")
    log.info(f"  Unidades: {resultado['unidades_totales']}")
    log.info("=" * 70)

    return resultado


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python etl_pipeline.py <ruta_archivo.xlsx|csv>")
        print("Ejemplo: python etl_pipeline.py data/ventas_dtf.xlsx")
        sys.exit(1)

    resultado = ejecutar_pipeline(sys.argv[1])
    print("\n✅ Resultado:")
    for k, v in resultado.items():
        print(f"  {k}: {v}")