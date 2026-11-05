"""Configuración central de BRUMA: rutas, fuentes de datos y umbrales."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"

TIMEZONE = "America/Bogota"

# --- Fuentes -----------------------------------------------------------------
# Red de calidad del aire del Área Metropolitana del Valle de Aburrá (operada por
# SIATA), publicada por IDEAM/SISAIRE en el portal de datos abiertos de Colombia.
SOCRATA_DATASET = "https://www.datos.gov.co/resource/g4t8-zkc3.csv"
SOCRATA_OPERATOR = "AMVA"
# Archivo de entrega propio de SIATA (ventana de ~1 año). Solo se usa para
# contrastar que ambas fuentes cuentan la misma historia.
SIATA_SNAPSHOT_URL = "https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_pm25.json"
# Reanálisis ERA5 vía Open-Meteo (sin API key).
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

# Punto meteorológico representativo del centro del valle (Medellín).
VALLEY_LAT, VALLEY_LON = 6.25, -75.58

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "cloud_cover",
    "boundary_layer_height",
    "surface_pressure",
]

# Estaciones con cobertura útil (>= ~70 % de horas) para el periodo 2020-2025.
MIN_STATION_HOURS = 30_000

# --- Calidad de datos ----------------------------------------------------------
PM25_MIN, PM25_MAX = 0.0, 500.0          # rango físico plausible (µg/m³)
SCALE_MONTH_MEDIAN = 60.0                 # mediana mensual imposible en el valle
SCALE_BAD_MONTHS = 2                      # meses sospechosos que invalidan el año
FLATLINE_HOURS = 6                        # >= 6 h con el mismo valor -> sensor pegado
MIN_HOURS_PER_DAY = 18                    # criterio del 75 % para un promedio de 24 h

# --- Índice de Calidad del Aire (Res. 2254 de 2017, PM2.5 promedio 24 h) -------
ICA_BANDS = [
    ("Buena", 0.0, 12.0, "#22c55e"),
    ("Aceptable", 12.0, 37.0, "#facc15"),
    ("Dañina a grupos sensibles", 37.0, 55.0, "#f97316"),
    ("Dañina a la salud", 55.0, 150.0, "#ef4444"),
    ("Muy dañina a la salud", 150.0, 250.0, "#a855f7"),
    ("Peligrosa", 250.0, 500.0, "#7f1d1d"),
]
ICA_NARANJA = 37.0   # umbral "naranja": promedio 24 h > 37 µg/m³

# La meteorología ERA5 del día objetivo equivale a un pronóstico del tiempo
# perfecto: no existe al momento de emitir. El modelo principal usa solo
# información disponible al cierre de D; la variante con ERA5 se reporta como
# experimento (ablación) en metrics.json.
USE_FUTURE_WEATHER = False

HORIZONS = (1, 2, 3)  # días hacia adelante (24, 48, 72 h)
QUANTILES = (0.1, 0.5, 0.9)

SEED = 42


def ica_category(value: float) -> dict:
    for name, lo, hi, color in ICA_BANDS:
        if value <= hi:
            return {"nombre": name, "color": color, "min": lo, "max": hi}
    name, lo, hi, color = ICA_BANDS[-1]
    return {"nombre": name, "color": color, "min": lo, "max": hi}
