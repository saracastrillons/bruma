"""Descarga de datos reales y caché local.

Cada función guarda la respuesta cruda en ``data/raw`` y la reutiliza si ya
existe, para que el pipeline sea reproducible sin volver a golpear las APIs.
"""
from __future__ import annotations

import io
import json
import time

import pandas as pd
import requests

from . import config

USER_AGENT = "BRUMA/1.0 (portafolio de ciencia de datos; github.com/saracastrillons)"


def _get(url: str, params: dict | None = None, retries: int = 4, timeout: int = 180) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            return r
        except requests.RequestException as exc:  # red inestable o throttling
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"No se pudo descargar {url}: {last}")


# --- PM2.5: red AMVA / SIATA vía datos.gov.co ----------------------------------

def fetch_station_catalog(force: bool = False) -> pd.DataFrame:
    """Estaciones AMVA que miden PM2.5, con coordenadas y número de horas."""
    path = config.RAW_DIR / "stations.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    params = {
        "$select": "estacion_id,nombre_est,municipio,latitud,longitud,altitud,count(*) AS n_horas,"
                   "min(med_fecha_inicio) AS desde,max(med_fecha_inicio) AS hasta",
        "$where": f"nombre_fgda='{config.SOCRATA_OPERATOR}' AND msfl_code='PM2.5'",
        "$group": "estacion_id,nombre_est,municipio,latitud,longitud,altitud",
        "$order": "n_horas DESC",
        "$limit": 500,
    }
    df = pd.read_csv(io.StringIO(_get(config.SOCRATA_DATASET, params).text))
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def fetch_station_pm25(station_id: int, force: bool = False, page: int = 50_000) -> pd.DataFrame:
    """Serie horaria cruda de PM2.5 de una estación (paginada)."""
    path = config.RAW_DIR / "pm25" / f"{station_id}.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    frames, offset = [], 0
    while True:
        params = {
            "$select": "estacion_id,med_fecha_inicio,med_concentracion_estandar",
            "$where": f"estacion_id='{station_id}' AND msfl_code='PM2.5' "
                      f"AND med_fecha_inicio between '{config.START_DATE}T00:00:00' "
                      f"and '{config.END_DATE}T23:59:59'",
            "$order": "med_fecha_inicio",
            "$limit": page,
            "$offset": offset,
        }
        chunk = pd.read_csv(io.StringIO(_get(config.SOCRATA_DATASET, params).text))
        frames.append(chunk)
        if len(chunk) < page:
            break
        offset += page
    df = pd.concat(frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# --- Meteorología: ERA5 vía Open-Meteo -----------------------------------------

def fetch_weather(force: bool = False) -> pd.DataFrame:
    """Meteorología horaria del centro del valle (hora local)."""
    path = config.RAW_DIR / "weather_era5.csv"
    if path.exists() and not force:
        return pd.read_csv(path, parse_dates=["time"])
    params = {
        "latitude": config.VALLEY_LAT,
        "longitude": config.VALLEY_LON,
        "start_date": config.START_DATE,
        "end_date": config.END_DATE,
        "hourly": ",".join(config.WEATHER_VARS),
        "timezone": config.TIMEZONE,
    }
    payload = _get(config.OPEN_METEO_ARCHIVE, params).json()
    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# --- Contraste: archivo propio de SIATA ----------------------------------------

def fetch_siata_snapshot(force: bool = False) -> pd.DataFrame | None:
    """Ventana de PM2.5 publicada directamente por SIATA. Devuelve None si no responde."""
    path = config.RAW_DIR / "siata_snapshot.json"
    if not path.exists() or force:
        try:
            raw = _get(config.SIATA_SNAPSHOT_URL, retries=2, timeout=90).content
        except RuntimeError:
            return None
        path.write_bytes(raw)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for st in data:
        for d in st["datos"]:
            rows.append((st["codigoSerial"], st["nombreCorto"], st["nombre"], st["latitud"],
                         st["longitud"], d["fecha"], d["valor"], d.get("calidad")))
    df = pd.DataFrame(rows, columns=["siata_id", "codigo", "nombre", "lat", "lon", "fecha", "pm25", "calidad"])
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df
