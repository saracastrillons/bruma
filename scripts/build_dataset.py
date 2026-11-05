"""Paso 2: limpia los datos crudos y genera los conjuntos procesados.

Salidas (data/processed):
  hourly_pm25.parquet   malla horaria limpia por estación
  daily_pm25.parquet    promedios de 24 h por estación
  daily_weather.parquet meteorología diaria del valle
  stations.csv          catálogo de estaciones usadas
Y artifacts/data_quality.json con el conteo de cada regla de limpieza.

Uso:  py scripts/build_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from bruma import clean, config, sources  # noqa: E402


def siata_crosscheck(hourly: pd.DataFrame, stations: pd.DataFrame) -> dict:
    """Compara datos.gov.co con el archivo propio de SIATA para la misma estación."""
    snap = sources.fetch_siata_snapshot()
    if snap is None:
        return {"disponible": False}
    snap = snap[snap["pm25"] > -9000].copy()
    # SIATA marca la hora con el FIN del intervalo; datos.gov.co con el INICIO.
    snap["time"] = snap["fecha"] - pd.Timedelta(hours=1)
    pairs = {"CEN-TRAF": 29509, "ITA-CJUS": 29493, "EST-HOSP": 31379,
             "MED-VILL": 31367, "MED-ARAN": 31364, "BEL-FEVE": 31362}
    raw = pd.concat([sources.fetch_station_pm25(s) for s in pairs.values()])
    raw = raw.assign(time=pd.to_datetime(raw["med_fecha_inicio"]))
    out = {"disponible": True, "desfase_horas": 1, "estaciones": {}}
    for code, sid in pairs.items():
        b = snap[snap["codigo"] == code].set_index("time")["pm25"]
        entry = {}
        for label, src in (("crudo", raw[raw["estacion_id"] == sid].set_index("time")["med_concentracion_estandar"]),
                           ("limpio", hourly[hourly["station"] == sid].set_index("time")["pm25"])):
            j = pd.concat([src, b], axis=1, join="inner").dropna()
            if len(j) > 10:
                entry[label] = {
                    "horas_comparadas": len(j),
                    "correlacion": round(float(j.corr().iloc[0, 1]), 4),
                    "error_abs_medio": round(float((j.iloc[:, 0] - j.iloc[:, 1]).abs().mean()), 3),
                }
        out["estaciones"][code] = entry
    return out


def main() -> None:
    catalog = sources.fetch_station_catalog()
    selected = catalog[catalog["n_horas"] >= config.MIN_STATION_HOURS].copy()
    raw = pd.concat([sources.fetch_station_pm25(int(s)) for s in selected["estacion_id"]], ignore_index=True)

    report: dict = {}
    hourly = clean.clean_hourly(raw, report)
    daily = clean.daily_means(hourly, report)
    weather_raw = sources.fetch_weather()
    missing = weather_raw.drop(columns="time").isna().sum()
    # p. ej. Open-Meteo no publica la altura de mezcla ERA5 de ene-jun 2024: se deja faltante
    report["meteo_horas_faltantes"] = {k: int(v) for k, v in missing[missing > 0].items()}
    weather = clean.daily_weather(weather_raw)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(config.PROCESSED_DIR / "hourly_pm25.parquet", index=False)
    daily.to_parquet(config.PROCESSED_DIR / "daily_pm25.parquet", index=False)
    weather.to_parquet(config.PROCESSED_DIR / "daily_weather.parquet", index=False)

    st = selected.rename(columns={"estacion_id": "station", "nombre_est": "nombre", "latitud": "lat",
                                  "longitud": "lon"})
    st["nombre"] = (st["nombre"].str.title()
                    .str.replace("I.E.", "I.E.", case=False, regex=False)
                    .str.replace("Epm", "EPM", regex=False)
                    .str.replace(r" B\.$", " Botero", regex=True))
    # nombres repetidos ("Hospital") -> se añade el municipio
    dup = st["nombre"].duplicated(keep=False) | (st["nombre"] == "Hospital")
    st.loc[dup, "nombre"] = st.loc[dup, "nombre"] + " " + st.loc[dup, "municipio"]
    valid_days = daily.dropna(subset=["pm25"]).groupby("station").size().rename("dias_validos")
    st = st.merge(valid_days, left_on="station", right_index=True, how="left")
    st[["station", "nombre", "municipio", "lat", "lon", "altitud", "n_horas", "dias_validos"]] \
        .to_csv(config.PROCESSED_DIR / "stations.csv", index=False)

    report["contraste_siata"] = siata_crosscheck(hourly, st)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.ARTIFACTS_DIR / "data_quality.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
