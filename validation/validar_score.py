"""
validation/validar_score.py — Validación empírica del Score de Viabilidad
═══════════════════════════════════════════════════════════════════════════════
Ejecuta el endpoint /analyze-design contra diseños históricos de la marca
y compara el score predicho vs el resultado real de ventas observado.

Output: CSV + tabla de correlación + análisis para incluir en la tesis.

Uso:
  1. Rellena el archivo disenos_historicos.csv con tus datos reales
  2. Coloca las imágenes en validation/imagenes/
  3. Ejecuta: python validation/validar_score.py
  4. Revisa validation/resultados_validacion.csv

Esto es oro académico: convierte el score de "heurística sin evidencia"
a "heurística validada empíricamente".
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("validar_score")

# Configuración
API_URL = os.getenv("API_URL", "http://localhost:8000")
CARPETA_IMAGENES = Path(__file__).parent / "imagenes"
CSV_INPUT = Path(__file__).parent / "disenos_historicos.csv"
CSV_OUTPUT = Path(__file__).parent / "resultados_validacion.csv"


def cargar_disenos_historicos() -> pd.DataFrame:
    """
    Lee el CSV con los diseños históricos a validar.

    Formato esperado:
      nombre_diseno,imagen,categoria,fecha_lanzamiento,piezas_vendidas,resultado_real

    resultado_real es la clasificación subjetiva del dueño:
      - EXITO   → se vendió bien (>umbral_alto piezas, o sold-out rápido)
      - MEDIO   → se vendió moderadamente (entre umbrales)
      - FRACASO → no se vendió (<umbral_bajo piezas, o tuvo que rebajarse)
    """
    if not CSV_INPUT.exists():
        log.error(f"No existe {CSV_INPUT}")
        log.info("Creando plantilla vacía...")
        crear_plantilla_vacia()
        log.info(f"✓ Plantilla creada: {CSV_INPUT}")
        log.info("  → Rellénala con tus datos reales y vuelve a ejecutar")
        sys.exit(0)

    df = pd.read_csv(CSV_INPUT)
    log.info(f"Cargados {len(df)} diseños históricos")
    return df


def crear_plantilla_vacia():
    """Genera un CSV de ejemplo que el usuario puede rellenar."""
    ejemplo = pd.DataFrame([
        {
            "nombre_diseno": "EJEMPLO_eliminar",
            "imagen": "aguila_mexicana.png",
            "categoria": "Casual",
            "fecha_lanzamiento": "2026-03-15",
            "piezas_vendidas": 8,
            "ingreso_bruto_mxn": 3200,
            "resultado_real": "EXITO",
            "notas": "Vendido en 2 semanas — esta fila es solo ejemplo, bórrala",
        },
        # Filas en blanco para que el usuario llene
        *[{
            "nombre_diseno": "",
            "imagen": "",
            "categoria": "",
            "fecha_lanzamiento": "",
            "piezas_vendidas": 0,
            "ingreso_bruto_mxn": 0,
            "resultado_real": "",
            "notas": "",
        } for _ in range(9)]
    ])
    CARPETA_IMAGENES.mkdir(exist_ok=True)
    ejemplo.to_csv(CSV_INPUT, index=False)


def evaluar_diseno(imagen_path: Path, categoria: str) -> dict:
    """Llama al endpoint /analyze-design y devuelve el resultado."""
    if not imagen_path.exists():
        return {"error": f"Imagen no encontrada: {imagen_path}"}

    mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime = mime_types.get(imagen_path.suffix.lower(), "image/jpeg")

    with open(imagen_path, "rb") as f:
        files = {"file": (imagen_path.name, f, mime)}
        data = {"geo": "MX", "categoria_override": categoria}
        try:
            r = requests.post(
                f"{API_URL}/api/v1/analyze-design",
                files=files,
                data=data,
                timeout=60,
            )
            if not r.ok:
                return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
            return r.json()
        except Exception as e:
            return {"error": str(e)}


def clasificar_score(score: float) -> str:
    """Convierte score numérico a clasificación categórica."""
    if score >= 70:
        return "EXITO_predicho"
    elif score >= 40:
        return "MEDIO_predicho"
    else:
        return "FRACASO_predicho"


def calcular_metricas(df: pd.DataFrame) -> dict:
    """
    Compara predicción vs realidad usando tres métricas:
      - Accuracy: % de diseños donde predicho == real
      - Confusion matrix simple
      - Correlación entre score numérico y piezas vendidas
    """
    # Mapeo para correlación
    mapa_real = {"FRACASO": 0, "MEDIO": 1, "EXITO": 2}
    mapa_pred = {"FRACASO_predicho": 0, "MEDIO_predicho": 1, "EXITO_predicho": 2}

    df_valido = df[df["score_predicho"].notna() & df["resultado_real"].isin(mapa_real.keys())].copy()

    if df_valido.empty:
        return {"error": "No hay datos válidos para métricas"}

    df_valido["real_num"] = df_valido["resultado_real"].map(mapa_real)
    df_valido["pred_num"] = df_valido["clasificacion_predicha"].map(mapa_pred)

    # Accuracy — misma clase
    accuracy = (df_valido["real_num"] == df_valido["pred_num"]).mean()

    # Accuracy relajado — a ±1 clase de distancia
    accuracy_relajado = (abs(df_valido["real_num"] - df_valido["pred_num"]) <= 1).mean()

    # Correlación score numérico vs piezas vendidas
    corr = df_valido[["score_predicho", "piezas_vendidas"]].corr().iloc[0, 1]

    # Matriz de confusión manual
    confusion = {}
    for real in ["FRACASO", "MEDIO", "EXITO"]:
        confusion[real] = {}
        for pred in ["FRACASO_predicho", "MEDIO_predicho", "EXITO_predicho"]:
            confusion[real][pred] = int(
                ((df_valido["resultado_real"] == real) &
                 (df_valido["clasificacion_predicha"] == pred)).sum()
            )

    return {
        "n_disenos_evaluados": len(df_valido),
        "accuracy_estricto": round(accuracy * 100, 1),
        "accuracy_relajado_±1": round(accuracy_relajado * 100, 1),
        "correlacion_score_vs_piezas_vendidas": round(float(corr), 3),
        "matriz_confusion": confusion,
    }


def main():
    log.info(f"Usando API: {API_URL}")
    df = cargar_disenos_historicos()

    # Filtrar filas vacías o de ejemplo
    df = df[
        df["nombre_diseno"].notna() &
        (df["nombre_diseno"] != "") &
        (df["nombre_diseno"] != "EJEMPLO_eliminar")
    ].copy()

    if df.empty:
        log.error("CSV está vacío. Rellénalo con diseños reales.")
        return

    log.info(f"Evaluando {len(df)} diseños...")

    # Evaluar cada diseño
    resultados = []
    for idx, row in df.iterrows():
        log.info(f"[{idx+1}/{len(df)}] {row['nombre_diseno']}...")
        imagen_path = CARPETA_IMAGENES / row["imagen"]

        resultado = evaluar_diseno(imagen_path, row["categoria"])

        if "error" in resultado:
            log.warning(f"  ✗ {resultado['error']}")
            resultados.append({
                **row.to_dict(),
                "score_predicho": None,
                "clasificacion_predicha": "ERROR",
                "keywords_detectadas": "",
                "componente_ml": None,
                "componente_trends": None,
                "componente_hm": None,
                "error": resultado["error"],
            })
            continue

        score = resultado["score_viabilidad"]
        componentes = resultado["componentes_score"]
        analisis = resultado["analisis_visual"]

        log.info(f"  ✓ score={score} ({clasificar_score(score)}) — "
                 f"ML:{componentes['demanda_ml']['score']:.0f} "
                 f"T:{componentes['tendencia_trends']['score']:.0f} "
                 f"H:{componentes['estacionalidad_hm']['score']:.0f}")

        resultados.append({
            **row.to_dict(),
            "score_predicho": score,
            "clasificacion_predicha": clasificar_score(score),
            "keywords_detectadas": ", ".join(analisis.get("keywords", [])),
            "componente_ml": componentes["demanda_ml"]["score"],
            "componente_trends": componentes["tendencia_trends"]["score"],
            "componente_hm": componentes["estacionalidad_hm"]["score"],
            "error": "",
        })

    # Guardar resultados
    df_out = pd.DataFrame(resultados)
    df_out.to_csv(CSV_OUTPUT, index=False)
    log.info(f"✓ Resultados guardados en {CSV_OUTPUT}")

    # Calcular y mostrar métricas
    print("\n" + "═" * 70)
    print("REPORTE DE VALIDACIÓN DEL SCORE DE VIABILIDAD")
    print("═" * 70)

    metricas = calcular_metricas(df_out)

    if "error" in metricas:
        print(f"\n⚠️  {metricas['error']}")
        print("   Rellena la columna 'resultado_real' con EXITO/MEDIO/FRACASO")
        return

    print(f"\nDiseños evaluados:        {metricas['n_disenos_evaluados']}")
    print(f"Accuracy estricto:        {metricas['accuracy_estricto']}%")
    print(f"Accuracy relajado (±1):   {metricas['accuracy_relajado_±1']}%")
    print(f"Correlación score↔ventas: {metricas['correlacion_score_vs_piezas_vendidas']}")

    print("\nMatriz de confusión:")
    print(f"{'':<10} {'FRACASO_p':<12} {'MEDIO_p':<12} {'EXITO_p':<12}")
    for real in ["FRACASO", "MEDIO", "EXITO"]:
        row = metricas["matriz_confusion"][real]
        print(f"{real:<10} {row['FRACASO_predicho']:<12} "
              f"{row['MEDIO_predicho']:<12} {row['EXITO_predicho']:<12}")

    # Interpretación
    print("\n" + "─" * 70)
    print("INTERPRETACIÓN PARA LA TESIS")
    print("─" * 70)

    acc = metricas["accuracy_estricto"]
    corr = metricas["correlacion_score_vs_piezas_vendidas"]

    if acc >= 60:
        print("✓ Accuracy >60%: el score tiene poder predictivo razonable.")
        print("  Defendible en tesis como 'heurística validada'.")
    elif acc >= 40:
        print("⚠ Accuracy 40-60%: mejor que azar (33%) pero marginal.")
        print("  Reportar honestamente y discutir limitaciones.")
    else:
        print("✗ Accuracy <40%: el score NO tiene poder predictivo.")
        print("  Opciones: (a) reenmarcar como 'filtro inicial', no predicción.")
        print("           (b) recalibrar pesos con grid search.")

    if abs(corr) >= 0.5:
        print(f"✓ Correlación |{corr}| alta: score correlaciona con ventas reales.")
    elif abs(corr) >= 0.3:
        print(f"◐ Correlación |{corr}| media: tendencia existe pero es débil.")
    else:
        print(f"✗ Correlación |{corr}| baja: score no correlaciona con ventas.")

    print()
    print("Guarda este output para el reporte técnico y la presentación.")


if __name__ == "__main__":
    main()