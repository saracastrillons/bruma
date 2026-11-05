"""Capa de datos del servicio: carga los artefactos y arma las respuestas."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import bulletin, config
from .features import FEATURE_LABELS

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]
ETIQUETAS = {1: "Mañana (24 h)", 2: "Pasado mañana (48 h)", 3: "En 3 días (72 h)"}


def format_value(feature: str, value, station_names: dict[int, str]) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "sin dato"
    if feature == "station":
        return station_names.get(int(value), str(value))
    if feature == "dow":
        return DIAS[int(value)]
    if feature == "month":
        return MESES[int(value) - 1]
    if feature in ("is_holiday", "is_weekend", "lockdown"):
        return "sí" if value else "no"
    if feature.startswith(("pm_", "valley_")):
        return f"{value:.1f} µg/m³"
    if feature.startswith("blh"):
        return f"{value:.0f} m"
    if feature.startswith(("precip", "rain_")):
        return f"{value:.1f} mm"
    if feature.startswith("temp"):
        return f"{value:.1f} °C"
    if feature.startswith(("rh_", "cloud_")):
        return f"{value:.0f} %"
    if feature.startswith(("wind_", "u_", "v_")):
        return f"{value:.1f} km/h"
    if feature.startswith("rad_"):
        return f"{value:.1f} kWh/m²"
    if feature.startswith("pressure"):
        return f"{value:.0f} hPa"
    return f"{value:.2f}"


class Store:
    def __init__(self, artifacts: Path = config.ARTIFACTS_DIR, processed: Path = config.PROCESSED_DIR):
        self.preds = pd.read_parquet(artifacts / "predictions.parquet")
        self.shap = pd.read_parquet(artifacts / "shap.parquet")
        self.feats = pd.read_parquet(artifacts / "features.parquet")
        if not (len(self.preds) == len(self.shap) == len(self.feats)):
            raise ValueError("predictions, shap y features deben tener las mismas filas")
        self.metrics = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))
        self.quality = json.loads((artifacts / "data_quality.json").read_text(encoding="utf-8"))
        self.daily = pd.read_parquet(processed / "daily_pm25.parquet")
        st = pd.read_csv(processed / "stations.csv")
        self.stations = st[st["station"].isin(self.preds["station"].unique())].reset_index(drop=True)
        self.names = dict(zip(self.stations["station"], self.stations["nombre"]))
        self.thresholds = {int(k): float(v) for k, v in self.metrics["umbral_alerta"].items()}
        self.preds["date"] = pd.to_datetime(self.preds["date"])
        self.preds["target_date"] = pd.to_datetime(self.preds["target_date"])
        self._by_key = {k: i for i, k in enumerate(zip(self.preds["station"], self.preds["date"], self.preds["horizon"]))}
        self.shap_cols = [c for c in self.shap.columns if c not in ("horizon", "base_value")]
        # fechas de emisión con pronóstico para todos los horizontes en al menos 10 estaciones
        h1 = self.preds[self.preds["horizon"] == 1]
        counts = h1.groupby("date")["station"].nunique()
        self.issue_dates = sorted(d for d, n in counts.items() if n >= 10)

    # --- consultas -----------------------------------------------------------------

    def meta(self) -> dict:
        return {
            "estaciones": self.stations[["station", "nombre", "municipio", "lat", "lon"]].to_dict("records"),
            "fechas": {"min": str(self.issue_dates[0].date()), "max": str(self.issue_dates[-1].date()),
                       "sugerida": self.suggested_date()},
            "umbral_naranja": config.ICA_NARANJA,
            "umbral_alerta": self.thresholds,
            "bandas_ica": [{"nombre": n, "min": lo, "max": hi, "color": c} for n, lo, hi, c in config.ICA_BANDS],
            "metricas": self.metrics_summary(),
        }

    def suggested_date(self) -> str:
        """Día con el mayor riesgo previsto: muestra un episodio real al abrir el tablero."""
        h1 = self.preds[self.preds["horizon"] == 1]
        return str(h1.groupby("date")["p_naranja"].max().idxmax().date())

    def resolve_date(self, date: str | None) -> pd.Timestamp:
        if not date:
            return pd.Timestamp(self.suggested_date())
        ts = pd.Timestamp(date)
        if ts < self.issue_dates[0] or ts > self.issue_dates[-1]:
            raise KeyError(f"Fecha fuera del periodo de evaluación ({self.issue_dates[0].date()} – "
                           f"{self.issue_dates[-1].date()})")
        # si ese día no hay emisión, usa la fecha disponible anterior más cercana
        i = int(np.searchsorted(np.array(self.issue_dates, dtype="datetime64[ns]"), ts.to_datetime64(), side="right")) - 1
        return self.issue_dates[max(i, 0)]

    def _row(self, station: int, date: pd.Timestamp, h: int) -> int | None:
        return self._by_key.get((station, date, h))

    def drivers(self, i: int, top: int = 6) -> dict:
        s = self.shap.iloc[i]
        f = self.feats.iloc[i]
        contrib = {c: float(s[c]) for c in self.shap_cols}
        order = sorted(contrib, key=lambda c: -abs(contrib[c]))
        items = [{
            "variable": c,
            "etiqueta": FEATURE_LABELS.get(c, c),
            "valor": format_value(c, f[c], self.names),
            "aporte": round(contrib[c], 2),
        } for c in order[:top]]
        rest = sum(contrib[c] for c in order[top:])
        return {
            "base": round(float(s["base_value"]), 2),
            "prediccion": round(float(s["base_value"]) + sum(contrib.values()), 2),
            "impulsores": items,
            "otros": round(rest, 2),
            "grupos": {k: round(v, 2) for k, v in bulletin.grouped_contributions(contrib).items()},
        }

    def forecast(self, station: int, date: str | None) -> dict:
        if station not in self.names:
            raise KeyError(f"Estación {station} desconocida")
        d = self.resolve_date(date)
        horizons = []
        for h in config.HORIZONS:
            i = self._row(station, d, h)
            if i is None:
                continue
            p = self.preds.iloc[i]
            horizons.append({
                "h": h,
                "etiqueta": ETIQUETAS[h],
                "fecha": str(p["target_date"].date()),
                "q10": round(float(p["q10"]), 1),
                "q50": round(float(p["q50"]), 1),
                "q90": round(float(p["q90"]), 1),
                "p_naranja": round(float(p["p_naranja"]), 3),
                "alerta": bool(p["p_naranja"] >= self.thresholds[h]),
                "ica": config.ica_category(float(p["q50"])),
                "observado": None if np.isnan(p["y"]) else round(float(p["y"]), 1),
                "persistencia": round(float(p["base_persistencia"]), 1),
                "explicacion": self.drivers(i),
            })
        if not horizons:
            raise KeyError("No hay pronóstico para esa estación en esa fecha")
        return {
            "estacion": {"id": station, "nombre": self.names[station]},
            "fecha_emision": str(d.date()),
            "horizontes": horizons,
            "historia": self.history(station, d),
        }

    def history(self, station: int, date: pd.Timestamp, back: int = 21, ahead: int = 3) -> list[dict]:
        s = self.daily[(self.daily["station"] == station)
                       & (self.daily["date"] > date - pd.Timedelta(days=back))
                       & (self.daily["date"] <= date + pd.Timedelta(days=ahead))]
        return [{"fecha": str(r.date.date()), "pm25": None if np.isnan(r.pm25) else round(float(r.pm25), 1),
                 "futuro": bool(r.date > date)} for r in s.itertuples()]

    def valley(self, date: str | None, h: int = 1) -> dict:
        d = self.resolve_date(date)
        sub = self.preds[(self.preds["date"] == d) & (self.preds["horizon"] == h)]
        rows = []
        for r in sub.itertuples():
            rows.append({
                "station": int(r.station), "nombre": self.names[int(r.station)],
                "q10": round(float(r.q10), 1), "q50": round(float(r.q50), 1), "q90": round(float(r.q90), 1),
                "p_naranja": round(float(r.p_naranja), 3), "alerta": bool(r.p_naranja >= self.thresholds[h]),
                "observado": None if np.isnan(r.y) else round(float(r.y), 1),
                "ica": config.ica_category(float(r.q50)),
            })
        st = self.stations.set_index("station")
        for row in rows:
            row["lat"], row["lon"] = float(st.loc[row["station"], "lat"]), float(st.loc[row["station"], "lon"])
        return {
            "fecha_emision": str(d.date()),
            "h": h,
            "fecha_objetivo": str((d + pd.Timedelta(days=h)).date()),
            "estaciones": rows,
            "promedio_valle": round(float(np.mean([r["q50"] for r in rows])), 1) if rows else None,
            "observado_valle": _mean_or_none([r["observado"] for r in rows]),
            "prob_max": max((r["p_naranja"] for r in rows), default=0.0),
            "estaciones_en_alerta": [r["nombre"] for r in rows if r["alerta"]],
        }

    def bulletin_facts(self, date: str | None) -> dict:
        d = self.resolve_date(date)
        horizons = []
        for h in config.HORIZONS:
            v = self.valley(str(d.date()), h)
            if not v["estaciones"]:
                continue
            idx = [self._row(r["station"], d, h) for r in v["estaciones"]]
            mean_contrib = self.shap.iloc[idx][self.shap_cols].mean().to_dict()
            lo = min(v["estaciones"], key=lambda r: r["q50"])
            hi = max(v["estaciones"], key=lambda r: r["q50"])
            horizons.append({
                "h": h,
                "etiqueta": ETIQUETAS[h],
                "fecha": v["fecha_objetivo"],
                "promedio_valle": v["promedio_valle"],
                "min": lo["q50"], "estacion_min": lo["nombre"],
                "max": hi["q50"], "estacion_max": hi["nombre"],
                "estaciones_en_alerta": v["estaciones_en_alerta"],
                "prob_max": round(v["prob_max"], 2),
                "impulsores": {k: round(val, 1) for k, val in
                               list(bulletin.grouped_contributions(mean_contrib).items())[:3]},
            })
        return bulletin.build_facts(str(d.date()), horizons)

    @lru_cache(maxsize=256)
    def bulletin(self, date: str) -> dict:
        return bulletin.make_bulletin(self.bulletin_facts(date))

    def metrics_summary(self) -> dict:
        out = {}
        for h, per in self.metrics["evaluacion"].items():
            m = per["2024-2025"]
            pt = m["puntual"]
            out[h] = {
                "n": m["n"],
                "mae": round(pt["LightGBM (BRUMA)"]["MAE"], 2),
                "mae_persistencia": round(pt["Persistencia (hoy = mañana)"]["MAE"], 2),
                "skill": round(pt["LightGBM (BRUMA)"]["skill_vs_persistencia"], 3),
                "r2": round(pt["LightGBM (BRUMA)"]["R2"], 3),
                "cobertura_80": round(m["intervalo_conformal"]["cobertura_80"], 3),
                "roc_auc": round(m["alerta_naranja"].get("roc_auc", float("nan")), 3),
                "pr_auc": round(m["alerta_naranja"].get("pr_auc", float("nan")), 3),
                "dias_naranja": m["alerta_naranja"]["n_dias_naranja"],
            }
        return out


def _mean_or_none(values):
    vals = [v for v in values if v is not None]
    return round(float(np.mean(vals)), 1) if vals else None
