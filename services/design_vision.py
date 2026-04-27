"""
services/design_vision.py — Análisis visual de diseños DTF
═══════════════════════════════════════════════════════════════════════════
Extrae keywords, categoría sugerida y atributos de diseño desde una imagen.

Soporta 3 backends (configurable vía VISION_PROVIDER en .env):
  • openai     → GPT-4o Vision  (requiere OPENAI_API_KEY)
  • anthropic  → Claude Vision  (requiere ANTHROPIC_API_KEY)
  • mock       → Stub determinístico para pruebas sin gastar API calls

Devuelve un dict normalizado:
{
    "keywords": ["streetwear", "skull", "y2k"],
    "categoria_sugerida": "Casual",
    "estilo": "minimalista",
    "colores_dominantes": ["negro", "rojo"],
    "audiencia_estimada": "jovenes 18-25",
    "complejidad": "media",  # baja | media | alta (para costo de impresión DTF)
    "raw_response": "..."     # útil para debug
}
═══════════════════════════════════════════════════════════════════════════
"""

import os
import json
import base64
import logging
from typing import Optional
from pathlib import Path

log = logging.getLogger("design_vision")

# Categorías válidas — DEBE coincidir con las del dataset DTF
CATEGORIAS_VALIDAS = [
    "Sports", "Gym", "Futbol", "Basketball", "Tenis", "Casual",
    "Movies", "Musica", "Hockey", "Deportiva", "Skateboarding",
    "Baseball", "Ufc", "Track and Field",
]

# Prompt que se envía al modelo de visión — versionado para reproducibilidad
PROMPT_VISION_V1 = f"""Analiza esta imagen de un diseño para playera DTF (Direct-to-Film).

Devuelve EXCLUSIVAMENTE un JSON válido (sin markdown, sin texto antes o después) con esta estructura:

{{
  "keywords": ["3 a 5 palabras clave en español o inglés que describan el diseño,
                útiles para buscar en Google Trends. Ejemplo: streetwear, anime, y2k"],
  "categoria_sugerida": "Una de estas opciones EXACTAS: {', '.join(CATEGORIAS_VALIDAS)}",
  "estilo": "minimalista | retro | urbano | deportivo | artistico | tipografico | ilustrado",
  "colores_dominantes": ["máximo 4 colores principales en español"],
  "audiencia_estimada": "descripción breve del público objetivo (edad, género, intereses)",
  "complejidad": "baja | media | alta"
}}

Criterios:
- "complejidad" baja = 1-2 colores planos, formas simples (más barato de imprimir DTF)
- "complejidad" alta = degradados, fotorrealismo, muchos detalles (más caro)
- Si no puedes identificar la categoría con confianza, usa "Casual" como default
- Las keywords deben ser sustantivos buscables, NO frases descriptivas
"""


# ═══════════════════════════════════════════════════════════════════════════
# BACKEND 1: MOCK (para pruebas sin API)
# ═══════════════════════════════════════════════════════════════════════════

def _analyze_mock(image_bytes: bytes) -> dict:
    """Stub determinístico basado en hash de la imagen — útil para tests."""
    import hashlib
    h = int(hashlib.md5(image_bytes).hexdigest(), 16)

    keywords_pool = [
        ["streetwear", "urbano", "minimalista"],
        ["anime", "manga", "japonés"],
        ["gym", "fitness", "motivacional"],
        ["futbol", "deportivo", "equipo"],
        ["retro", "vintage", "y2k"],
        ["skater", "punk", "grunge"],
    ]
    categorias_pool = ["Casual", "Gym", "Sports", "Futbol", "Skateboarding", "Musica"]
    estilos = ["minimalista", "urbano", "deportivo", "retro"]
    complejidades = ["baja", "media", "alta"]

    return {
        "keywords": keywords_pool[h % len(keywords_pool)],
        "categoria_sugerida": categorias_pool[h % len(categorias_pool)],
        "estilo": estilos[h % len(estilos)],
        "colores_dominantes": ["negro", "blanco"],
        "audiencia_estimada": "jovenes 18-30",
        "complejidad": complejidades[h % len(complejidades)],
        "raw_response": "[MOCK] Análisis simulado — configura VISION_PROVIDER en .env",
    }


# ═══════════════════════════════════════════════════════════════════════════
# BACKEND 2: OPENAI GPT-4o VISION
# ═══════════════════════════════════════════════════════════════════════════

def _analyze_openai(image_bytes: bytes) -> dict:
    """Llama a GPT-4o Vision. Requiere OPENAI_API_KEY en el entorno."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("Instala openai: pip install openai>=1.0.0")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY en variables de entorno")

    client = OpenAI(api_key=api_key)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # mini es 10x más barato y suficiente para esto
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_VISION_V1},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        }],
        max_tokens=400,
        temperature=0.2,  # bajo para JSON consistente
    )

    raw = response.choices[0].message.content.strip()
    return _parse_vision_response(raw)


# ═══════════════════════════════════════════════════════════════════════════
# BACKEND 3: ANTHROPIC CLAUDE VISION
# ═══════════════════════════════════════════════════════════════════════════

def _analyze_anthropic(image_bytes: bytes) -> dict:
    """Llama a Claude Vision. Requiere ANTHROPIC_API_KEY en el entorno."""
    try:
        from anthropic import Anthropic
    except ImportError:
        raise RuntimeError("Instala anthropic: pip install anthropic>=0.40.0")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Falta ANTHROPIC_API_KEY en variables de entorno")

    client = Anthropic(api_key=api_key)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Haiku es rápido y barato
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": PROMPT_VISION_V1},
            ],
        }],
    )

    raw = response.content[0].text.strip()
    return _parse_vision_response(raw)


# ═══════════════════════════════════════════════════════════════════════════
# PARSER + DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════

def _parse_vision_response(raw: str) -> dict:
    """Limpia markdown fences y parsea JSON. Resiliente a respuestas sucias."""
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.error(f"Vision devolvió JSON inválido: {raw[:200]}")
        raise RuntimeError(f"Respuesta de Vision no es JSON válido: {e}")

    # Validación de campos obligatorios
    required = ["keywords", "categoria_sugerida", "estilo", "complejidad"]
    for field in required:
        if field not in parsed:
            raise RuntimeError(f"Campo faltante en respuesta de Vision: {field}")

    # Normalizar categoría sugerida al catálogo permitido
    cat = parsed["categoria_sugerida"]
    if cat not in CATEGORIAS_VALIDAS:
        log.warning(f"Categoría '{cat}' no válida, usando 'Casual'")
        parsed["categoria_sugerida"] = "Casual"

    # Limitar keywords a 5 (límite de Google Trends)
    parsed["keywords"] = parsed["keywords"][:5]

    parsed["raw_response"] = raw
    return parsed


def analyze_design(image_bytes: bytes, provider: Optional[str] = None) -> dict:
    """
    Analiza un diseño DTF y extrae keywords + metadata.

    Args:
        image_bytes: contenido binario de la imagen (jpg/png)
        provider: 'openai', 'anthropic', 'mock' o None (lee VISION_PROVIDER)

    Returns:
        dict con keywords, categoria_sugerida, estilo, complejidad, etc.
    """
    provider = provider or os.getenv("VISION_PROVIDER", "mock").lower()

    log.info(f"Analizando diseño con provider='{provider}' ({len(image_bytes)} bytes)")

    if provider == "openai":
        return _analyze_openai(image_bytes)
    elif provider == "anthropic":
        return _analyze_anthropic(image_bytes)
    elif provider == "mock":
        return _analyze_mock(image_bytes)
    else:
        raise ValueError(f"VISION_PROVIDER desconocido: {provider}")