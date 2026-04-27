"""
Diagnóstico del error HTTP 404 en /api/v1/analyze-design
═══════════════════════════════════════════════════════════════════════════════
Ejecuta esto para identificar EXACTAMENTE qué está mal.

Uso:
    docker-compose exec api python /diagnostico_404.py

O desde fuera del contenedor:
    python diagnostico_404.py --url http://localhost:8000
"""

import sys
import argparse
import requests


def check(label: str, condition: bool, detail: str = ""):
    """Imprime resultado de un check."""
    icon = "✓" if condition else "✗"
    color = "\033[92m" if condition else "\033[91m"
    reset = "\033[0m"
    print(f"  {color}{icon}{reset} {label}{(' — ' + detail) if detail else ''}")
    return condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000",
                        help="URL del backend FastAPI")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    print(f"\n{'═' * 70}")
    print(f"DIAGNÓSTICO HTTP 404 — {base}")
    print(f"{'═' * 70}\n")

    # ───────────────────────────────────────────────────────────────────
    # Test 1: ¿El backend está corriendo?
    # ───────────────────────────────────────────────────────────────────
    print("1. Conectividad básica")
    try:
        r = requests.get(f"{base}/", timeout=5)
        check("Backend responde a /", r.ok, f"HTTP {r.status_code}")
    except requests.exceptions.ConnectionError:
        check("Backend responde a /", False, "Connection refused — ¿está corriendo?")
        print("\n  → Solución: docker-compose up -d")
        return
    except Exception as e:
        check("Backend responde a /", False, str(e))
        return

    # ───────────────────────────────────────────────────────────────────
    # Test 2: ¿Swagger /docs funciona?
    # ───────────────────────────────────────────────────────────────────
    print("\n2. Documentación Swagger")
    try:
        r = requests.get(f"{base}/docs", timeout=5)
        check("/docs accesible", r.ok, f"HTTP {r.status_code}")
    except Exception as e:
        check("/docs accesible", False, str(e))

    # ───────────────────────────────────────────────────────────────────
    # Test 3: ¿OpenAPI schema lista todos los endpoints?
    # ───────────────────────────────────────────────────────────────────
    print("\n3. Endpoints registrados en la API")
    try:
        r = requests.get(f"{base}/openapi.json", timeout=5)
        if not r.ok:
            check("OpenAPI schema disponible", False, f"HTTP {r.status_code}")
            return

        schema = r.json()
        paths = list(schema.get("paths", {}).keys())

        # Endpoints críticos que deben existir
        endpoints_esperados = {
            "/predict": "Predicción base (debe existir desde v4.0)",
            "/ingest": "Ingesta individual (debe existir desde v4.1)",
            "/upload": "Carga de archivo Excel/CSV",
            "/api/v1/analyze-design": "🎨 Analizador de Diseños (v5.0+)",
            "/api/v1/categorias-disponibles": "Soporte del Analizador",
            "/api/v1/analyze-theme": "🌟 Explorador de Temas (v5.2+)",
        }

        for endpoint, descripcion in endpoints_esperados.items():
            existe = endpoint in paths
            check(f"{endpoint}", existe, descripcion)

        if "/api/v1/analyze-design" not in paths:
            print()
            print("  ⚠️  CAUSA RAÍZ DEL 404 IDENTIFICADA:")
            print("     El endpoint /api/v1/analyze-design NO está registrado en la API.")
            print()
            print("  Esto significa que api/main.py NO incluye el router de design.")
            print()
            print("  Verifica que api/main.py tenga:")
            print("     from api.routes_design import router as design_router")
            print("     app.include_router(design_router)")
            print()

    except Exception as e:
        check("OpenAPI schema disponible", False, str(e))

    # ───────────────────────────────────────────────────────────────────
    # Test 4: ¿GET en endpoint de soporte funciona?
    # ───────────────────────────────────────────────────────────────────
    print("\n4. Endpoints de soporte")
    try:
        r = requests.get(f"{base}/api/v1/categorias-disponibles", timeout=5)
        if r.status_code == 404:
            check("GET /categorias-disponibles", False, "404 — router no incluido")
        elif r.ok:
            data = r.json()
            check("GET /categorias-disponibles", True,
                  f"{data.get('total', 0)} categorías disponibles")
        else:
            check("GET /categorias-disponibles", False, f"HTTP {r.status_code}")
    except Exception as e:
        check("GET /categorias-disponibles", False, str(e))

    # ───────────────────────────────────────────────────────────────────
    # Test 5: Variables de entorno críticas
    # ───────────────────────────────────────────────────────────────────
    print("\n5. Variables de entorno (verifica desde dentro del contenedor)")
    print("  Para verificar ejecuta:")
    print("    docker-compose exec api env | grep -E 'VISION|OPENAI|DATABASE'")
    print()
    print("  Variables esperadas:")
    print("    VISION_PROVIDER=openai")
    print("    OPENAI_API_KEY=sk-proj-...")
    print("    DATABASE_URL=postgresql+psycopg2://...")

    print(f"\n{'═' * 70}\n")


if __name__ == "__main__":
    main()