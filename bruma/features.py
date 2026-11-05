"""Construcción de la matriz de entrenamiento.

Cada fila es (estación, día de emisión D, horizonte h). El pronóstico se emite al
cierre del día D usando solo información disponible hasta las 23:00 de D, y el
objetivo es el promedio de 24 h de PM2.5 del día D+h.

La meteorología del día objetivo proviene del reanálisis ERA5: es un *sustituto*
de un pronóstico numérico del tiempo (en operación real se usaría el pronóstico
de Open-Meteo). Por eso la evaluación también reporta un modelo sin meteorología
futura, como cota inferior honesta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as _holidays
except ImportError:  # pragma: no cover - dependencia opcional
    _holidays = None

WEATHER_COLS = [
    "temp_mean", "temp_range", "rh_mean", "precip_sum", "wind_mean", "u_mean", "v_mean",
    "rad_sum", "cloud_mean", "blh_day", "blh_night", "pressure_mean",
]
# Subconjunto de meteorología del día objetivo (sustituto de pronóstico).
TARGET_WEATHER_COLS = ["precip_sum", "rh_mean", "blh_day", "blh_night", "wind_mean", "rad_sum",
                       "cloud_mean", "temp_range", "u_mean", "v_mean"]

LOCKDOWN = (pd.Timestamp("2020-03-20"), pd.Timestamp("2020-08-31"))


def colombian_holidays(years) -> set[pd.Timestamp]:
    if _holidays is None:
        return set()
    return {pd.Timestamp(d) for d in _holidays.country_holidays("CO", years=list(years))}


def build_panel(daily_pm: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Panel diario completo (todas las fechas x estaciones) con variables rezagadas.

    Todas las variables de este panel describen el día de emisión D o anteriores."""
    dates = pd.date_range(daily_pm["date"].min(), daily_pm["date"].max(), freq="D")
    stations = sorted(daily_pm["station"].unique())
    idx = pd.MultiIndex.from_product([stations, dates], names=["station", "date"])
    p = daily_pm.set_index(["station", "date"]).reindex(idx).reset_index()
    p = p.sort_values(["station", "date"])

    g = p.groupby("station")["pm25"]
    p["pm_d0"] = p["pm25"]
    for k in (1, 2, 6):
        p[f"pm_d{k}"] = g.shift(k)
    p["pm_mean3"] = g.transform(lambda s: s.rolling(3, min_periods=2).mean())
    p["pm_mean7"] = g.transform(lambda s: s.rolling(7, min_periods=4).mean())
    p["pm_mean28"] = g.transform(lambda s: s.rolling(28, min_periods=14).mean())
    p["pm_std7"] = g.transform(lambda s: s.rolling(7, min_periods=4).std())
    p["pm_trend"] = p["pm_d0"] - p["pm_d1"]
    p["pm_evening"] = p["pm25_evening"]

    # Señal regional: promedio de todas las estaciones del valle el día D.
    valley = p.groupby("date")["pm25"].mean().rename("valley_d0")
    p = p.merge(valley, on="date", how="left")
    p["valley_mean7"] = p.groupby("station")["valley_d0"].transform(lambda s: s.rolling(7, min_periods=4).mean())

    w = weather.set_index("date")[WEATHER_COLS]
    p = p.merge(w.add_suffix("_d0"), left_on="date", right_index=True, how="left")
    rain3 = w["precip_sum"].rolling(3, min_periods=1).sum().rename("precip_3d")
    p = p.merge(rain3, left_on="date", right_index=True, how="left")
    return p


def make_horizon_frame(panel: pd.DataFrame, weather: pd.DataFrame, horizon: int,
                       future_weather: bool = True) -> pd.DataFrame:
    """Filas (estación, D) con el objetivo y(D+h) y las variables del día objetivo."""
    df = panel.copy()
    df["target_date"] = df["date"] + pd.Timedelta(days=horizon)
    target = panel[["station", "date", "pm25"]].rename(columns={"date": "target_date", "pm25": "y"})
    df = df.merge(target, on=["station", "target_date"], how="left")

    # Variable de referencia para la línea base "misma semana": y(D+h-7)
    seasonal = panel[["station", "date", "pm25"]].rename(columns={"pm25": "pm_same_dow_last_week"})
    seasonal["date"] = seasonal["date"] + pd.Timedelta(days=7 - horizon)
    df = df.merge(seasonal, on=["station", "date"], how="left")

    t = df["target_date"]
    hol = colombian_holidays(range(t.dt.year.min(), t.dt.year.max() + 1))
    df["dow"] = t.dt.dayofweek
    df["month"] = t.dt.month
    doy = t.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["is_holiday"] = t.isin(hol).astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["lockdown"] = t.between(*LOCKDOWN).astype(int)

    if future_weather:
        w = weather.set_index("date")[TARGET_WEATHER_COLS].add_suffix("_t")
        df = df.merge(w, left_on="target_date", right_index=True, how="left")
        df["blh_change"] = df["blh_day_t"] - df["blh_day_d0"]
        df["rain_window"] = weather.set_index("date")["precip_sum"].rolling(horizon, min_periods=1).sum() \
            .reindex(df["target_date"]).to_numpy()
    return df


def feature_columns(future_weather: bool = True) -> list[str]:
    cols = [
        "station", "pm_d0", "pm_d1", "pm_d2", "pm_d6", "pm_mean3", "pm_mean7", "pm_mean28", "pm_std7",
        "pm_trend", "pm_evening", "valley_d0", "valley_mean7", "precip_3d",
        *[f"{c}_d0" for c in WEATHER_COLS],
        "dow", "month", "doy_sin", "doy_cos", "is_holiday", "is_weekend", "lockdown",
    ]
    if future_weather:
        cols += [f"{c}_t" for c in TARGET_WEATHER_COLS] + ["blh_change", "rain_window"]
    return cols


# Nombres legibles en español para el tablero y el boletín.
FEATURE_LABELS = {
    "station": "Estación (nivel típico del sitio)",
    "pm_d0": "PM2.5 de hoy",
    "pm_d1": "PM2.5 de ayer",
    "pm_d2": "PM2.5 de hace 2 días",
    "pm_d6": "PM2.5 de hace 6 días",
    "pm_mean3": "Promedio PM2.5 últimos 3 días",
    "pm_mean7": "Promedio PM2.5 última semana",
    "pm_mean28": "Promedio PM2.5 último mes",
    "pm_std7": "Variabilidad PM2.5 semana",
    "pm_trend": "Tendencia hoy vs ayer",
    "pm_evening": "PM2.5 de esta noche (18-23 h)",
    "valley_d0": "PM2.5 promedio del valle hoy",
    "valley_mean7": "PM2.5 del valle última semana",
    "precip_3d": "Lluvia acumulada 3 días",
    "dow": "Día de la semana",
    "month": "Mes",
    "doy_sin": "Época del año (sen)",
    "doy_cos": "Época del año (cos)",
    "is_holiday": "Festivo",
    "is_weekend": "Fin de semana",
    "lockdown": "Cuarentena 2020",
    "blh_change": "Cambio en altura de mezcla",
    "rain_window": "Lluvia hasta el día objetivo",
}
_W = {
    "temp_mean": "Temperatura media", "temp_range": "Amplitud térmica", "rh_mean": "Humedad relativa",
    "precip_sum": "Lluvia", "wind_mean": "Velocidad del viento", "u_mean": "Viento este-oeste",
    "v_mean": "Viento norte-sur", "rad_sum": "Radiación solar", "cloud_mean": "Nubosidad",
    "blh_day": "Altura de mezcla (día)", "blh_night": "Altura de mezcla (noche)",
    "pressure_mean": "Presión",
}
for _k, _v in _W.items():
    FEATURE_LABELS[f"{_k}_d0"] = f"{_v} hoy"
    FEATURE_LABELS[f"{_k}_t"] = f"{_v} del día pronosticado"
