# Plataforma de Análisis Predictivo con IA para Gestión de Demanda en Marcas de Moda DTF

> Proyecto de titulación — Enero 2026 – Mayo 2026  
> Santiago Tapia & Fernando Flores

Plataforma de análisis predictivo basada en inteligencia artificial para pronosticar la demanda de diseños y productos impresos bajo tecnología **Direct-to-Film (DTF)**. El sistema integra datos históricos de ventas, transferencia de patrones estacionales desde el dataset H&M (31.7M transacciones), y señales externas (Google Trends) para generar pronósticos accionables que optimicen las decisiones de producción e inventario.

---

## Problema

Las marcas de moda que operan con producción DTF de bajo volumen enfrentan:

- **Sobreproducción** — se imprimen diseños que no se venden, generando pérdidas.
- **Desabasto (stockouts)** — diseños populares se agotan por falta de anticipación.
- **Decisiones intuitivas** — la producción se basa en experiencia, no en datos.
- **Oportunidades perdidas** — tendencias emergentes se detectan demasiado tarde.

La gestión reactiva basada en análisis histórico descriptivo resulta insuficiente frente al dinamismo del mercado de moda digital.

---

## Solución

Un pipeline end-to-end que combina dos enfoques complementarios:

1. **Modelos ML independientes** (Random Forest, SARIMA) entrenados sobre features de la serie temporal de ventas DTF.
2. **Modelo de transferencia estadística** que extrae patrones estacionales macro del dataset H&M (31.7M transacciones) y los calibra al volumen real de DTF Fashion.

Los modelos se exponen mediante una API REST y un dashboard interactivo.

### Resultados obtenidos

| Modelo | MAPE | Mejora vs Baseline |
|--------|------|--------------------|
| Baseline Naive | 53.04% | — |
| SARIMA(1,1,1)(1,1,1)[7] | — | — |
| **Random Forest** | **32.74%** | **+38.3%** |
| Transferencia H&M | 15.48% (sobre serie H&M) | — |

> **Nota:** Las métricas de SARIMA se calculan al ejecutar `train_models.py` con los datos reales. El modelo Random Forest supera el objetivo de mejora del 20% con un **+38.3%** de reducción en MAPE frente al baseline naive.

### Modelo de Transferencia H&M

| Indicador | Valor |
|-----------|-------|
| Dataset de referencia | H&M Personalized Fashion (Kaggle) |
| Transacciones procesadas | 31,788,324 |
| Modelo SARIMA aplicado | SARIMA(1,1,1)(1,1,1)[7] sobre serie limpia |
| MAPE del modelo H&M | 15.48% |
| Outliers corregidos en H&M | 26 días (IQR ×2.5) |
| Pronóstico DTF a 30 días | 61 unidades (rango: 43–80) |

> El modelo híbrido SARIMA+XGBoost fue probado y **descartado** porque empeoró el MAPE de 16.27% a 23.43%. XGBoost amplificó outliers al sobreajustar al patrón de fin de semana. Los modelos ML se usan de forma independiente, NO como correctores de residuos SARIMA. Ver Reporte Técnico Final, Sección 2.5.

---

## Arquitectura

```
┌──────────────────┐     ┌──────────────────┐     ┌─────────────────────┐     ┌───────────────┐
│  Fuentes de Datos│────▶│  Módulo ETL v2.0  │────▶│  Entrenamiento ML   │────▶│  API FastAPI   │
│                  │     │  Python/Pandas    │     │                     │     │  v2.0          │
│ • Excel Ventas   │     │ • Ingesta/merge   │     │ • SARIMA real       │     │ /predict       │
│ • Índices H&M    │     │ • Limpieza        │     │ • Random Forest     │     │ /transfer      │
│ • Google Trends  │     │ • Feature Eng.    │     │ • XGBoost real      │     │ /trends/live   │
│                  │     │ • Outlier detect  │     │ • Transfer H&M      │     │ /seasonality   │
│                  │     │ • H&M calibration │     │ • Ensemble óptimo   │     │ /recommendations│
└──────────────────┘     └──────────────────┘     └─────────────────────┘     └───────┬───────┘
                                                                                      │
                                                                             ┌────────▼────────┐
                                                                             │   Dashboard     │
                                                                             │   Streamlit v2.0│
                                                                             │                 │
                                                                             │ • Pronósticos   │
                                                                             │ • Rankings      │
                                                                             │ • H&M Transfer  │
                                                                             │ • Google Trends │
                                                                             │ • Producción    │
                                                                             └─────────────────┘
```

---

## Estructura del repositorio

```
dtf-predictive-platform/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── .gitignore
│
├── data/
│   ├── raw/                              # Datos originales (no en git)
│   │   └── DTF_s_DATA_CORRECT.xlsx
│   └── processed/                        # Datos limpios generados por ETL
│       ├── ventas_limpias.csv
│       ├── serie_semanal.csv
│       ├── serie_semanal_limpia.csv      # (v2.0) Outliers suavizados
│       ├── features_modelo.csv
│       ├── baseline_naive.csv
│       ├── resumen_eda.csv
│       ├── factores_correccion_hm.csv    # (v2.0) Factores H&M → DTF
│       ├── indices_estacionales_hm.json  # (v2.0) Índices H&M hardcodeados
│       └── pipeline_metadata.json
│
├── etl/
│   └── etl_pipeline.py                   # Pipeline ETL v2.0
│
├── models/
│   ├── train_models.py                   # Entrenamiento v2.0 (SARIMA + RF + XGBoost + H&M)
│   └── saved/                            # Modelos serializados (.pkl)
│       ├── random_forest_model.pkl
│       ├── xgboost_model.pkl             # (v2.0) Antes: gradient_boosting_model.pkl
│       ├── comparacion_modelos.csv
│       ├── feature_importance.csv
│       ├── predicciones_completas.csv
│       ├── sarima_diagnostico.json       # (v2.0) Diagnóstico SARIMA por categoría
│       └── training_report.json
│
├── api/
│   └── main.py                           # API REST FastAPI v2.0
│
├── dashboard/
│   └── app.py                            # Dashboard Streamlit v2.0
│
├── research/                             # Experimentos de investigación H&M
│   ├── README.md                         # Documentación de los experimentos
│   ├── hm_limpieza_agregacion.py         # Limpieza dataset H&M (3.5GB)
│   ├── hm_exploracion.py                 # Análisis exploratorio H&M
│   ├── hm_sarima.py                      # SARIMA base — iteración 1
│   ├── hm_xgboost.py                     # Híbrido SARIMA+XGBoost — descartado
│   ├── hm_sarima_mejorado.py             # SARIMA con limpieza outliers — final
│   ├── dtf_finetuning.py                 # Calibración de patrones H&M → DTF
│   └── demo_trends.py                    # Demo de integración pytrends
│
├── docs/
│   ├── DTF_Fashion_Reporte_Tecnico_Final.pdf
│   └── Predictive_Analysis_Platform_for_Fashion_Brands.pdf
│
└── notebooks/
    └── eda_exploratorio.ipynb
```

---

## Instalación

### Prerrequisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (incluye Docker Engine y Docker Compose)
- Git

### Setup con Docker

```bash
# 1. Clonar repositorio
git clone https://github.com/mrbugs16/DTF-Predictive-Platform.git
cd DTF-Predictive-Platform

# 2. Crear archivo de variables de entorno
cp .env.example .env
# Edita .env si necesitas cambiar contraseñas o puertos

# 3. Levantar todos los servicios (base de datos + API + dashboard)
docker-compose up -d

# 4. Verificar que los contenedores estén corriendo
docker-compose ps
```

Los servicios estarán disponibles en:

| Servicio | URL |
|----------|-----|
| Dashboard | http://localhost:8501 |
| API REST | http://localhost:8000 |
| Docs interactivos (Swagger) | http://localhost:8000/docs |

### Comandos útiles

```bash
# Ver logs en tiempo real
docker-compose logs -f

# Ver logs de un servicio específico
docker-compose logs -f api
docker-compose logs -f dashboard

# Reconstruir imágenes después de cambios en el código
docker-compose up -d --build

# Detener todos los servicios (conserva los datos de la BD)
docker-compose down

# Detener y borrar todos los datos de la BD (reset completo)
docker-compose down -v
```

### Entrenamiento de modelos

El entrenamiento inicial carga los datos y genera los archivos `.pkl` que usa la API:

```bash
docker-compose exec api python models/train_models.py
```

> **Nota:** El archivo de datos originales (`data/raw/DTF_s_DATA_CORRECT.xlsx`) no está en el repositorio. Colócalo en esa ruta antes de ejecutar el entrenamiento.

---

## API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/` | Health check y estado del sistema |
| `GET` | `/predict/{categoria}` | Pronóstico por categoría con bandas ±30% |
| `GET` | `/predict` | Pronóstico de todas las categorías |
| `GET` | `/transfer/{categoria}` | Pronóstico modelo transferencia H&M **(v2.0)** |
| `GET` | `/trends` | Ranking de categorías por demanda predicha |
| `GET` | `/trends/live` | Google Trends en tiempo real **(v2.0)** |
| `GET` | `/metrics` | Métricas de evaluación (5 modelos + ensemble) |
| `GET` | `/features` | Feature importance (incluye features H&M) |
| `GET` | `/recommendations` | Recomendaciones de producción con escenarios |
| `GET` | `/seasonality` | Índices estacionales H&M vs DTF **(v2.0)** |
| `GET` | `/history/{categoria}` | Historial de ventas semanales |
| `POST` | `/train` | Re-entrenar modelos con datos actualizados |

### Ejemplo de uso

```bash
# Obtener pronóstico de la categoría Gym (incluye bandas de incertidumbre)
curl http://localhost:8000/predict/Gym

# Obtener pronóstico con modelo de transferencia H&M
curl http://localhost:8000/transfer/Gym

# Consultar Google Trends para una categoría
curl "http://localhost:8000/trends/live?categoria=Futbol&timeframe=today%203-m"

# Comparar estacionalidad H&M vs DTF
curl http://localhost:8000/seasonality

# Obtener recomendaciones de producción con escenarios
curl http://localhost:8000/recommendations
```

---

## Modelos de Machine Learning

### 1. SARIMA(1,1,1)(1,1,1)[7]
Modelo de series temporales con componente estacional semanal (s=7). Parámetros validados por análisis ACF/PACF sobre el dataset H&M. Usa `statsmodels.SARIMAX` con fallback a Holt-Winters para categorías con pocas observaciones (<14 semanas).

### 2. Random Forest
Ensemble de árboles de decisión que expone la importancia relativa de cada feature. Robusto frente a outliers y datos faltantes.

### 3. Transferencia H&M (v2.0)
Modelo puramente estadístico que no usa ML. Extrae índices estacionales normalizados del dataset H&M (31.7M transacciones), los escala al volumen DTF, y los corrige mes a mes con datos reales de la tienda. Su valor está en capturar patrones macro de la industria de moda que los modelos ML no pueden aprender con solo 54 transacciones locales.

### Iteración descartada: SARIMA + XGBoost híbrido
Se probó usar XGBoost como corrector de residuos del SARIMA (13 features de calendario y rezagos). El MAPE empeoró de 16.27% a 23.43% porque XGBoost sobreajustó al patrón de fin de semana (`es_finde` importancia: 0.20) y amplificó un día atípico. Se descartó el enfoque híbrido. Ver `research/hm_xgboost.py`.

### Features predictivos principales

| Feature | Tipo | Interpretación |
|---------|------|----------------|
| `cambio_semanal` | Lag | Aceleración de demanda — el mejor predictor |
| `lag_1w` | Lag | Ventas de la semana pasada |
| `ventas_acumuladas` | Acumulado | Popularidad total de la categoría |
| `rolling_mean_2w` | Rolling | Promedio móvil de 2 semanas |
| `hm_idx_mensual` | H&M Transfer | Índice estacional mensual H&M |
| `hm_pred_escalada` | H&M Transfer | Predicción H&M calibrada a escala DTF |
| `ratio_vs_hm_lag1` | H&M Transfer | Desviación DTF vs H&M la semana pasada |
| `trends_score` | Google Trends | Interés de búsqueda (placeholder) |

---

## Datos

El dataset actual contiene **54 transacciones** de una marca DTF operando desde octubre 2025, con las siguientes dimensiones:

- **13 categorías**: Sports, Gym, Fútbol, Basketball, Tenis, Casual, Movies, Música, etc.
- **4 tipos de prenda**: T-Shirt (69%), Jacket (13%), Hoodie (9%), Long Sleeve (9%)
- **14 estados** de México + 1 internacional (Canadá)
- **37 diseños únicos**

### Métricas financieras del período

| Indicador | Valor |
|-----------|-------|
| Período de análisis | Octubre 2025 – Marzo 2026 |
| Ingresos brutos totales | $24,092.00 MXN |
| Ingreso neto total | $19,623.48 MXN |
| Margen neto promedio | 81.5% |
| Ticket promedio | $446.15 MXN |
| Categoría líder por volumen | Sports — 14 unidades (25.9%) |
| Categoría líder por ingreso | Gym — $5,846 MXN |
| Concentración geográfica | CDMX — 47% de las ventas |

### Hallazgos clave

- **Rotación mensual de categorías dominantes**: Fútbol en diciembre, Gym en enero-febrero, Sports en marzo.
- **Bursts de demanda**: Track & Field representó 14 de 16 ventas en marzo, todas desde CDMX.
- **Divergencia estacional H&M vs DTF**: DTF muestra pico los domingos (compradores online), H&M los sábados (tienda física). Febrero y marzo en DTF superan la predicción H&M en 53% y 75% respectivamente.

---

## Transferencia de patrones H&M

### ¿Por qué se usa el dataset H&M?

Con solo 54 transacciones, entrenar SARIMA directamente sobre los datos DTF genera overfitting severo. Se adoptó una estrategia de **transferencia de conocimiento estadístico** usando el dataset público H&M (Kaggle, 2022) como prior:

1. Se procesaron 31.7M de transacciones para extraer índices estacionales normalizados (semanal y mensual).
2. Se limpió la serie con IQR ×2.5 (26 outliers corregidos), mejorando el MAPE de 16.27% a 15.48%.
3. Los índices se escalaron al volumen DTF con un factor de escala de 0.0000346.
4. Se calcularon factores de corrección mensual para ajustar divergencias culturales (mercado europeo vs mexicano).

Los scripts de este proceso están en la carpeta `research/` y el reporte técnico detallado en `docs/`.

---

## Deploy

La plataforma está containerizada con Docker Compose (tres servicios: PostgreSQL, FastAPI, Streamlit). Cualquier servidor con Docker instalado puede levantarla con `docker-compose up -d`.

### Railway / Render (nube)

Ambas plataformas soportan despliegue directo desde Docker Compose. Configura las variables de entorno del `.env` como secrets en el panel de la plataforma.

### Dominio propio

Apunta el subdominio al servidor y usa un proxy inverso (nginx o Caddy) para exponer los puertos 8000 (API) y 8501 (dashboard).

---

## Metodología

El desarrollo sigue la metodología **APQP** (Advanced Product Quality Planning) adaptada a software:

| Fase | Periodo | Descripción |
|------|---------|-------------|
| 1. Planeación | Ene 12 – Mar 14 | Definición del problema, requerimientos, stack |
| 2. Diseño | Mar 15 – Mar 29 | Arquitectura, pipeline ETL, diseño de API |
| 3. Implementación | Mar 28 – Abr 21 | ETL, modelos ML, API, dashboard, transfer H&M |
| 4. Validación | Abr 20 – Abr 27 | Métricas, pruebas end-to-end, comparación modelos |
| 5. Cierre | Abr 28 – May 4 | Documentación, presentación oral |

---

## Stack tecnológico

| Componente | Tecnología |
|------------|------------|
| Lenguaje | Python 3.10+ |
| Base de datos | PostgreSQL 16 |
| Infraestructura | Docker, Docker Compose |
| ETL | pandas, numpy |
| ML | scikit-learn, statsmodels (SARIMAX) |
| API | FastAPI, uvicorn, pydantic |
| Dashboard | Streamlit, Plotly |
| Señales externas | pytrends (Google Trends) |
| Dataset de referencia | H&M Kaggle (31.7M transacciones) |
| Serialización | joblib, JSON |
| Versionado | Git / GitHub |

---

## Métricas de evaluación

- **MAE** (Mean Absolute Error): error promedio en unidades. Menor = mejor.
- **MAPE** (Mean Absolute Percentage Error): error porcentual. Menor = mejor.
- **R²** (Coeficiente de determinación): varianza explicada. Mayor = mejor (máximo 1.0).

El objetivo del proyecto es una **mejora mínima del 20%** en MAPE respecto al baseline naive. El resultado actual del mejor modelo individual es **+38.3%** (Random Forest, MAPE 32.74%).

---

## Trabajo futuro

- Integración completa de **Google Trends API** como feature activo en el pipeline ETL (actualmente es placeholder).
- **Sistema de recomendación** tipo collaborative filtering ("clientes que compraron X también compraron Y").
- **MLflow** para versionado y tracking de experimentos de modelos.
- **Piloto de validación** de 4 semanas para medir MAPE real contra el baseline empírico.

---

## Autores

Santiago Tapia & Fernando Flores

## Licencia

Uso académico. Todos los derechos reservados.