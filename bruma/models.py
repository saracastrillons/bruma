"""Modelos, líneas base y evaluación temporal sin fuga de información.

Esquema de validación (origen móvil, por años):

* Pliegues de ajuste (2022, 2023): solo se usan para elegir hiperparámetros y
  el umbral de alerta. Cada uno entrena con objetivos estrictamente anteriores.
* Pliegues de evaluación (2024, 2025): nunca se miran durante el ajuste. El
  modelo de 2024 entrena con datos < 2024; el de 2025 con datos < 2025.

Para cada horizonte h (1, 2, 3 días) se entrena un modelo directo:
  - LightGBM cuantil 0.1 / 0.5 / 0.9  -> pronóstico central + banda del 80 %
  - LightGBM binario                  -> P(promedio 24 h > 37 µg/m³, ICA naranja)
La banda se recalibra con predicción conformal (CQR) usando los residuos fuera
de muestra de los años anteriores.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import feature_columns

warnings.filterwarnings("ignore", message="X does not have valid feature names")

TUNING_YEARS = (2022, 2023)
EVAL_YEARS = (2024, 2025)

BASE_PARAMS = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                   random_state=config.SEED, verbose=-1, n_jobs=-1)
PARAM_GRID = [
    dict(num_leaves=7, min_child_samples=50, learning_rate=0.05, n_estimators=400),
    dict(num_leaves=15, min_child_samples=30, learning_rate=0.03, n_estimators=500),
    dict(num_leaves=15, min_child_samples=30, learning_rate=0.03, n_estimators=900),
    dict(num_leaves=31, min_child_samples=20, learning_rate=0.03, n_estimators=500),
]


# --- utilidades ------------------------------------------------------------------

def prepare_X(df: pd.DataFrame, cols: list[str], stations: list[int]) -> pd.DataFrame:
    X = df[cols].copy()
    X["station"] = pd.Categorical(X["station"], categories=stations)
    return X


def split_year(df: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrena con objetivos < 1-ene-year; prueba con objetivos dentro de year."""
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year + 1}-01-01")
    train = df[df["target_date"] < start]
    test = df[(df["target_date"] >= start) & (df["target_date"] < end)]
    return train, test


def pinball(y, q, alpha):
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


# --- líneas base -----------------------------------------------------------------

def climatology(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Media de entrenamiento por estación y mes del día objetivo (respaldo: mes)."""
    by_sm = train.groupby(["station", "month"])["y"].mean()
    by_m = train.groupby("month")["y"].mean()
    keys = pd.MultiIndex.from_arrays([test["station"], test["month"]])
    out = by_sm.reindex(keys).to_numpy()
    fallback = by_m.reindex(test["month"]).to_numpy()
    return np.where(np.isnan(out), fallback, out)


def exceed_climatology(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    freq = (train["y"] > config.ICA_NARANJA).groupby(train["month"]).mean()
    return freq.reindex(test["month"]).fillna(0).to_numpy()


def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    pers = test["pm_d0"].to_numpy()
    # si hoy no hay dato, la persistencia usa el último promedio semanal
    pers = np.where(np.isnan(pers), test["pm_mean7"].to_numpy(), pers)
    clim = climatology(train, test)
    pers = np.where(np.isnan(pers), clim, pers)
    seas = test["pm_same_dow_last_week"].to_numpy()
    seas = np.where(np.isnan(seas), pers, seas)
    return {"persistencia": pers, "estacional_semanal": seas, "climatologia": clim}


# --- modelos ---------------------------------------------------------------------

@dataclass
class HorizonModels:
    horizon: int
    cols: list[str]
    stations: list[int]
    quantile: dict[float, lgb.LGBMRegressor] = field(default_factory=dict)
    classifier: lgb.LGBMClassifier | None = None
    conformal_offset: float = 0.0

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = prepare_X(df, self.cols, self.stations)
        out = pd.DataFrame(index=df.index)
        preds = {q: m.predict(X) for q, m in self.quantile.items()}
        lo, mid, hi = (preds[q] for q in config.QUANTILES)
        # garantiza cuantiles monótonos y aplica la corrección conformal
        stack = np.sort(np.vstack([lo, mid, hi]), axis=0)
        out["q10_raw"], out["q50"], out["q90_raw"] = stack
        out["q10"] = np.clip(out["q10_raw"] - self.conformal_offset, 0, None)
        out["q90"] = out["q90_raw"] + self.conformal_offset
        out["p_naranja"] = self.classifier.predict_proba(X)[:, 1] if self.classifier else np.nan
        return out

    def shap_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """TreeSHAP exacto del modelo mediano (contribuciones en µg/m³)."""
        X = prepare_X(df, self.cols, self.stations)
        contrib = self.quantile[0.5].booster_.predict(X, pred_contrib=True)
        return pd.DataFrame(contrib, columns=[*self.cols, "base_value"], index=df.index)


def fit_horizon(train: pd.DataFrame, horizon: int, params: dict, stations: list[int],
                future_weather: bool = True, with_classifier: bool = True) -> HorizonModels:
    cols = feature_columns(future_weather)
    X = prepare_X(train, cols, stations)
    y = train["y"].to_numpy()
    hm = HorizonModels(horizon=horizon, cols=cols, stations=stations)
    for q in config.QUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS, **params)
        m.fit(X, y)
        hm.quantile[q] = m
    if with_classifier:
        yb = (y > config.ICA_NARANJA).astype(int)
        clf = lgb.LGBMClassifier(objective="binary", **BASE_PARAMS, **params)
        clf.fit(X, yb)
        hm.classifier = clf
    return hm


def fit_ridge(train: pd.DataFrame, test: pd.DataFrame, future_weather: bool = True) -> np.ndarray:
    cols = [c for c in feature_columns(future_weather) if c != "station"]
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[cols], train["y"])
    return model.predict(test[cols])


# --- métricas --------------------------------------------------------------------

def point_metrics(y: np.ndarray, pred: np.ndarray, ref_mae: float | None = None) -> dict:
    err = pred - y
    mae = float(np.mean(np.abs(err)))
    out = {
        "MAE": mae,
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "sesgo": float(np.mean(err)),
        "R2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
    }
    if ref_mae:
        out["skill_vs_persistencia"] = 1 - mae / ref_mae
    return out


def interval_metrics(y, lo, hi, q50) -> dict:
    return {
        "cobertura_80": float(np.mean((y >= lo) & (y <= hi))),
        "ancho_medio": float(np.mean(hi - lo)),
        "pinball_medio": float(np.mean([pinball(y, lo, 0.1), pinball(y, q50, 0.5), pinball(y, hi, 0.9)])),
    }


def exceedance_metrics(y: np.ndarray, p: np.ndarray, p_ref: np.ndarray, threshold: float) -> dict:
    yb = (y > config.ICA_NARANJA).astype(int)
    out = {"n_dias_naranja": int(yb.sum()), "n": int(len(yb))}
    out["brier"] = float(brier_score_loss(yb, p))
    ref = float(brier_score_loss(yb, p_ref))
    out["brier_skill_vs_climatologia"] = 1 - out["brier"] / ref if ref > 0 else None
    if 0 < yb.sum() < len(yb):
        out["roc_auc"] = float(roc_auc_score(yb, p))
        out["pr_auc"] = float(average_precision_score(yb, p))
        out["tasa_base"] = float(yb.mean())
    pred = p >= threshold
    tp = int((pred & (yb == 1)).sum())
    out.update({
        "umbral": threshold,
        "verdaderos_positivos": tp,
        "falsas_alarmas": int((pred & (yb == 0)).sum()),
        "eventos_perdidos": int((~pred & (yb == 1)).sum()),
        "recall": tp / yb.sum() if yb.sum() else None,
        "precision": tp / pred.sum() if pred.sum() else None,
    })
    return out


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    yb = (y > config.ICA_NARANJA).astype(int)
    best, best_t = -1.0, 0.5
    for t in np.arange(0.05, 0.95, 0.025):
        pred = p >= t
        tp = (pred & (yb == 1)).sum()
        if pred.sum() == 0 or yb.sum() == 0:
            continue
        prec, rec = tp / pred.sum(), tp / yb.sum()
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        if f1 > best:
            best, best_t = f1, float(t)
    return round(best_t, 3)


def conformal_offset(y, lo, hi, alpha: float = 0.2) -> float:
    """Desplazamiento CQR (Romano et al., 2019) a partir de residuos fuera de muestra."""
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level))
