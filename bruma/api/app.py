"""API de BRUMA (FastAPI) + tablero estático.

Ejecutar:  py -m uvicorn bruma.api.app:app --reload
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import config
from ..service import Store

load_dotenv(config.ROOT / ".env")

STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="BRUMA · API de pronóstico de PM2.5",
    description="Pronósticos fuera de muestra (2024-2025) de PM2.5 a 24-72 h para el Valle de Aburrá, "
                "con bandas de incertidumbre, probabilidad de nivel naranja y explicaciones SHAP.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def store() -> Store:
    return Store()


def _guard(fn, *args):
    try:
        return fn(*args)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\"")) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/healthz")
def healthz():
    s = store()
    return {"ok": True, "filas": len(s.preds), "estaciones": len(s.names)}


@app.get("/api/meta")
def meta():
    return store().meta()


@app.get("/api/forecast")
def forecast(station: int, date: str | None = Query(None, description="Fecha de emisión AAAA-MM-DD")):
    """Pronóstico 24/48/72 h de una estación, con banda del 80 %, P(naranja) y explicación SHAP."""
    return _guard(store().forecast, station, date)


@app.get("/api/valley")
def valley(date: str | None = None, h: int = Query(1, ge=1, le=3)):
    """Pronóstico de todas las estaciones para un horizonte (para el mapa)."""
    return _guard(store().valley, date, h)


@app.get("/api/bulletin")
def bulletin(date: str | None = None):
    """Boletín en lenguaje natural generado SOLO con salidas del modelo."""
    s = store()
    d = _guard(s.resolve_date, date)
    return s.bulletin(str(d.date()))


@app.get("/api/metrics")
def metrics():
    s = store()
    return {"resumen": s.metrics_summary(), "completo": s.metrics, "calidad_datos": s.quality}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
