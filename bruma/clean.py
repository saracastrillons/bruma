"""Limpieza y agregación diaria.

Reglas (cada una se cuenta en ``artifacts/data_quality.json``):

0. Periodos con escala inconsistente (ver ``scale_inconsistent_mask``) -> faltante.
1. Valores no numéricos o fuera de rango físico (``<0`` o ``>500 µg/m³``) -> faltante.
2. Horas duplicadas para la misma estación -> se promedian.
3. Sensor "pegado": ``FLATLINE_HOURS`` o más horas seguidas con el mismo valor -> faltante.
4. Picos aislados: una hora > 100 µg/m³ que supera 4x la mediana de sus ±3 h vecinas -> faltante.
5. Promedio diario solo si hay al menos ``MIN_HOURS_PER_DAY`` horas válidas (criterio del 75 %).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def clean_hourly(raw: pd.DataFrame, report: dict | None = None) -> pd.DataFrame:
    """Recibe filas crudas (estacion_id, med_fecha_inicio, med_concentracion_estandar)
    de una o varias estaciones y devuelve una malla horaria limpia ``station, time, pm25``."""
    report = report if report is not None else {}
    df = pd.DataFrame({
        "station": raw["estacion_id"].astype(int),
        "time": pd.to_datetime(raw["med_fecha_inicio"]),
        "pm25": pd.to_numeric(raw["med_concentracion_estandar"], errors="coerce"),
    })
    _count(report, "filas_crudas", len(df))

    scale = scale_inconsistent_mask(df)
    _count(report, "periodo_escala_inconsistente", int(scale.sum()))
    df.loc[scale, "pm25"] = np.nan

    bad =df["pm25"].isna() | (df["pm25"] < config.PM25_MIN) | (df["pm25"] > config.PM25_MAX)
    _count(report, "fuera_de_rango_o_no_numerico", int((bad & ~scale).sum()))
    df.loc[bad, "pm25"] = np.nan

    n_before = len(df)
    df = df.groupby(["station", "time"], as_index=False)["pm25"].mean()
    _count(report, "horas_duplicadas", n_before - len(df))

    out = []
    for station, g in df.groupby("station"):
        g = g.set_index("time").sort_index()
        grid = pd.date_range(g.index.min().floor("D"), g.index.max().ceil("D") - pd.Timedelta(hours=1), freq="h")
        s = g["pm25"].reindex(grid)

        flat = _flatline_mask(s, config.FLATLINE_HOURS)
        _count(report, "sensor_pegado", int(flat.sum()))
        s[flat] = np.nan

        spikes = _spike_mask(s)
        _count(report, "picos_aislados", int(spikes.sum()))
        s[spikes] = np.nan

        out.append(pd.DataFrame({"station": station, "time": s.index, "pm25": s.to_numpy()}))
    clean = pd.concat(out, ignore_index=True)
    _count(report, "horas_validas", int(clean["pm25"].notna().sum()))
    return clean


def scale_inconsistent_mask(df: pd.DataFrame) -> pd.Series:
    """Detecta periodos en los que una estación no reporta en µg/m³.

    En datos.gov.co, 14 estaciones de la red AMVA publican entre 2020 y 2023
    valores con medianas mensuales de 100-2000 (físicamente imposibles como
    PM2.5 en el valle); desde 2024 coinciden exactamente con el archivo de SIATA.
    Regla: un mes-estación es sospechoso si su mediana > ``SCALE_MONTH_MEDIAN``
    o si más del 2 % de sus horas superan 500. Si un año tiene
    ``SCALE_BAD_MONTHS`` o más meses sospechosos, se descarta el año completo
    (el problema es de régimen, no de horas aisladas)."""
    month = df["time"].dt.to_period("M")
    g = df.groupby(["station", month])["pm25"]
    stats = pd.DataFrame({"median": g.median(), "frac_big": g.apply(lambda s: (s > 500).mean())})
    flagged = (stats["median"] > config.SCALE_MONTH_MEDIAN) | (stats["frac_big"] > 0.02)
    flagged_df = flagged.rename("flag").reset_index()
    flagged_df["year"] = flagged_df["time"].dt.year
    bad_years = flagged_df.groupby(["station", "year"])["flag"].sum() >= config.SCALE_BAD_MONTHS

    key_m = pd.MultiIndex.from_arrays([df["station"], month])
    key_y = pd.MultiIndex.from_arrays([df["station"], df["time"].dt.year])
    mask = flagged.reindex(key_m).to_numpy(dtype=bool) | bad_years.reindex(key_y).fillna(False).to_numpy(dtype=bool)
    return pd.Series(mask, index=df.index)


def _flatline_mask(s: pd.Series, min_run: int) -> pd.Series:
    """True en horas que pertenecen a una racha de ``min_run`` o más valores idénticos."""
    v = s.to_numpy()
    same = np.r_[False, (v[1:] == v[:-1]) & ~np.isnan(v[1:])]
    run_id = np.cumsum(~same)
    run_len = pd.Series(run_id).map(pd.Series(run_id).value_counts()).to_numpy()
    return pd.Series((run_len >= min_run) & ~np.isnan(v), index=s.index)


def _spike_mask(s: pd.Series) -> pd.Series:
    neigh = pd.concat([s.shift(k) for k in (-3, -2, -1, 1, 2, 3)], axis=1).median(axis=1)
    return (s > 100) & (s > 4 * neigh)


def daily_means(hourly: pd.DataFrame, report: dict | None = None) -> pd.DataFrame:
    """Promedio diario (24 h) por estación + promedio de las horas 18-23 (cierre del día)."""
    h = hourly.copy()
    h["date"] = h["time"].dt.normalize()
    h["evening"] = h["pm25"].where(h["time"].dt.hour >= 18)
    agg = h.groupby(["station", "date"]).agg(
        pm25=("pm25", "mean"), n_hours=("pm25", "count"), pm25_evening=("evening", "mean"),
        pm25_max=("pm25", "max"),
    ).reset_index()
    insufficient = agg["n_hours"] < config.MIN_HOURS_PER_DAY
    if report is not None:
        _count(report, "dias_estacion_total", len(agg))
        _count(report, "dias_descartados_menos_de_18h", int(insufficient.sum()))
    agg.loc[insufficient, ["pm25", "pm25_max"]] = np.nan
    return agg


def daily_weather(hourly: pd.DataFrame) -> pd.DataFrame:
    """Agrega la meteorología horaria a variables diarias interpretables."""
    w = hourly.copy()
    w["date"] = pd.to_datetime(w["time"]).dt.normalize()
    rad = np.deg2rad(w["wind_direction_10m"])
    # componentes del viento (hacia dónde sopla): u>0 este, v>0 norte
    w["u"] = -w["wind_speed_10m"] * np.sin(rad)
    w["v"] = -w["wind_speed_10m"] * np.cos(rad)
    day = w["time"].dt.hour.between(10, 16)
    w["blh_day"] = w["boundary_layer_height"].where(day)
    w["blh_night"] = w["boundary_layer_height"].where(~w["time"].dt.hour.between(7, 19))
    g = w.groupby("date")
    out = pd.DataFrame({
        "temp_mean": g["temperature_2m"].mean(),
        "temp_range": g["temperature_2m"].max() - g["temperature_2m"].min(),
        "rh_mean": g["relative_humidity_2m"].mean(),
        "precip_sum": g["precipitation"].sum(),
        "wind_mean": g["wind_speed_10m"].mean(),
        "u_mean": g["u"].mean(),
        "v_mean": g["v"].mean(),
        "rad_sum": g["shortwave_radiation"].sum() / 1000.0,   # kWh/m²
        "cloud_mean": g["cloud_cover"].mean(),
        "blh_day": g["blh_day"].mean(),
        "blh_night": g["blh_night"].mean(),
        "pressure_mean": g["surface_pressure"].mean(),
    })
    return out.reset_index()


def _count(report: dict, key: str, n: int) -> None:
    report[key] = report.get(key, 0) + int(n)
