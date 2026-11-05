"""Paso 1: descarga y cachea los datos crudos.

Uso:  py scripts/download_data.py [--force]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bruma import config, sources  # noqa: E402


def main(force: bool = False) -> None:
    catalog = sources.fetch_station_catalog(force=force)
    selected = catalog[catalog["n_horas"] >= config.MIN_STATION_HOURS]
    print(f"Estaciones AMVA con PM2.5: {len(catalog)} · seleccionadas: {len(selected)}")
    for _, st in selected.iterrows():
        df = sources.fetch_station_pm25(int(st["estacion_id"]), force=force)
        print(f"  {st['estacion_id']:>6} {st['nombre_est'][:38]:<38} {len(df):>6} horas")
    weather = sources.fetch_weather(force=force)
    print(f"Meteorología ERA5: {len(weather)} horas")
    snap = sources.fetch_siata_snapshot(force=force)
    print("Archivo SIATA:", "no disponible" if snap is None else f"{len(snap)} registros")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
