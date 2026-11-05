"""Paso 3: ajuste, validación temporal, evaluación y explicaciones.

Salidas (artifacts/):
  metrics.json          tabla completa de métricas (ajuste + evaluación)
  predictions.parquet   pronósticos fuera de muestra 2024-2025 + líneas base
  shap.parquet          contribuciones SHAP del modelo mediano (mismas filas)
  features.parquet      valores de las variables (mismas filas)
  importance.json       importancia global (media |SHAP|) por horizonte
  models/               modelos finales entrenados con todo 2020-2025

Uso:  py scripts/train.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bruma import config, models  # noqa: E402
from bruma.features import build_panel, feature_columns, make_horizon_frame  # noqa: E402

MAIN_FW = config.USE_FUTURE_WEATHER
ALT_LABEL = ("LightGBM sin meteorología del día objetivo" if MAIN_FW
             else "LightGBM + meteorología ERA5 del día objetivo")


def load():
    daily = pd.read_parquet(config.PROCESSED_DIR / "daily_pm25.parquet")
    weather = pd.read_parquet(config.PROCESSED_DIR / "daily_weather.parquet")
    return daily, weather


def tune(frames: dict[int, pd.DataFrame], stations: list[int]) -> tuple[dict, list]:
    """Elige hiperparámetros por MAE medio del modelo mediano en los pliegues de ajuste."""
    results = []
    for i, params in enumerate(models.PARAM_GRID):
        maes = []
        for h, df in frames.items():
            for year in models.TUNING_YEARS:
                train, test = models.split_year(df, year)
                hm = models.fit_horizon(train, h, params, stations, future_weather=MAIN_FW,
                                        with_classifier=False)
                maes.append(np.mean(np.abs(hm.predict(test)["q50"] - test["y"])))
        results.append({"config": i, **params, "MAE_cv": float(np.mean(maes))})
        print(f"  config {i}: MAE CV = {results[-1]['MAE_cv']:.3f}")
    best = min(results, key=lambda r: r["MAE_cv"])
    return {k: best[k] for k in models.PARAM_GRID[0]}, results


def out_of_sample(df: pd.DataFrame, df_alt: pd.DataFrame, h: int, params: dict, stations: list[int]):
    """Predicciones de cada año con un modelo entrenado solo con años anteriores."""
    rows, shap_rows, fitted = [], [], {}
    for year in (*models.TUNING_YEARS, *models.EVAL_YEARS):
        train, test = models.split_year(df, year)
        hm = models.fit_horizon(train, h, params, stations, future_weather=MAIN_FW)
        pred = hm.predict(test)
        base = models.baseline_predictions(train, test)
        tr_alt, te_alt = models.split_year(df_alt, year)
        alt = models.fit_horizon(tr_alt, h, params, stations, future_weather=not MAIN_FW, with_classifier=False)
        out = test[["station", "date", "target_date", "y", "month"]].copy()
        out["horizon"] = h
        out = out.join(pred)
        out["q50_variante"] = alt.predict(te_alt)["q50"].to_numpy()
        out["ridge"] = models.fit_ridge(train, test, future_weather=MAIN_FW)
        for k, v in base.items():
            out[f"base_{k}"] = v
        out["p_climatologia"] = models.exceed_climatology(train, test)
        rows.append(out)
        if year in models.EVAL_YEARS:
            sv = hm.shap_values(test)
            shap_rows.append(sv.astype("float32"))
        fitted[year] = hm
    return pd.concat(rows), pd.concat(shap_rows), fitted


def summarize(oos: pd.DataFrame, thresholds: dict[int, float]) -> dict:
    out = {}
    for h, g in oos.groupby("horizon"):
        per = {}
        for label, sub in [*((str(y), g[g["target_date"].dt.year == y]) for y in models.EVAL_YEARS),
                           ("2024-2025", g[g["target_date"].dt.year.isin(models.EVAL_YEARS)])]:
            y = sub["y"].to_numpy()
            ref = float(np.mean(np.abs(sub["base_persistencia"] - y)))
            point = {
                "LightGBM (BRUMA)": models.point_metrics(y, sub["q50"].to_numpy(), ref),
                ALT_LABEL: models.point_metrics(y, sub["q50_variante"].to_numpy(), ref),
                "Regresión Ridge": models.point_metrics(y, sub["ridge"].to_numpy(), ref),
                "Persistencia (hoy = mañana)": models.point_metrics(y, sub["base_persistencia"].to_numpy(), ref),
                "Estacional semanal": models.point_metrics(y, sub["base_estacional_semanal"].to_numpy(), ref),
                "Climatología estación-mes": models.point_metrics(y, sub["base_climatologia"].to_numpy(), ref),
            }
            season = sub["target_date"].dt.month.isin([2, 3, 4])
            per[label] = {
                "n": int(len(sub)),
                "puntual": point,
                "intervalo_crudo": models.interval_metrics(y, sub["q10_raw"], sub["q90_raw"], sub["q50"]),
                "intervalo_conformal": models.interval_metrics(y, sub["q10"], sub["q90"], sub["q50"]),
                "alerta_naranja": models.exceedance_metrics(y, sub["p_naranja"].to_numpy(),
                                                            sub["p_climatologia"].to_numpy(), thresholds[h]),
                "MAE_feb_abr": float(np.mean(np.abs(sub.loc[season, "q50"] - sub.loc[season, "y"]))),
                "MAE_resto": float(np.mean(np.abs(sub.loc[~season, "q50"] - sub.loc[~season, "y"]))),
                "MAE_persistencia_feb_abr": float(np.mean(np.abs(sub.loc[season, "base_persistencia"] - sub.loc[season, "y"]))),
            }
        out[f"h{h}"] = per
    return out


def main() -> None:
    t0 = time.time()
    daily, weather = load()
    stations = sorted(int(s) for s in daily["station"].unique())
    panel = build_panel(daily, weather)

    frames, frames_alt = {}, {}
    for h in config.HORIZONS:
        frames[h] = make_horizon_frame(panel, weather, h, future_weather=MAIN_FW) \
            .dropna(subset=["y"]).reset_index(drop=True)
        frames_alt[h] = make_horizon_frame(panel, weather, h, future_weather=not MAIN_FW) \
            .dropna(subset=["y"]).reset_index(drop=True)
        print(f"h={h}: {len(frames[h])} filas, {len(feature_columns(MAIN_FW))} variables")

    print("Ajuste de hiperparámetros (pliegues 2022, 2023)…")
    params, grid = tune(frames, stations)
    print("  elegido:", params)

    oos_all, shap_all, thresholds, offsets = [], [], {}, {}
    for h in config.HORIZONS:
        oos, shap, _ = out_of_sample(frames[h], frames_alt[h], h, params, stations)
        # umbral de alerta: mejor F1 en los pliegues de ajuste (sin mirar 2024-2025)
        tun = oos[oos["target_date"].dt.year.isin(models.TUNING_YEARS)]
        thresholds[h] = models.best_f1_threshold(tun["y"].to_numpy(), tun["p_naranja"].to_numpy())
        # conformal: el desplazamiento de cada año sale de los residuos de años anteriores
        offsets[h] = {}
        for year in models.EVAL_YEARS:
            prev = oos[oos["target_date"].dt.year < year]
            off = models.conformal_offset(prev["y"].to_numpy(), prev["q10_raw"].to_numpy(), prev["q90_raw"].to_numpy())
            m = oos["target_date"].dt.year == year
            oos.loc[m, "q10"] = np.clip(oos.loc[m, "q10_raw"] - off, 0, None)
            oos.loc[m, "q90"] = oos.loc[m, "q90_raw"] + off
            offsets[h][year] = round(off, 3)
        oos_all.append(oos)
        shap_all.append(shap.assign(horizon=h))
        print(f"h={h}: umbral alerta={thresholds[h]}, conformal={offsets[h]}")

    oos = pd.concat(oos_all)
    metrics = {
        "descripcion": "Evaluación fuera de muestra con origen móvil por años. Hiperparámetros y umbral "
                       "elegidos solo con 2022-2023; 2024 y 2025 nunca se usaron para ajustar.",
        "hiperparametros": params,
        "busqueda": grid,
        "umbral_alerta": thresholds,
        "desplazamiento_conformal": offsets,
        "evaluacion": summarize(oos, thresholds),
        "ajuste": summarize_tuning(oos),
    }

    eval_mask = oos["target_date"].dt.year.isin(models.EVAL_YEARS)
    preds = oos[eval_mask].reset_index()
    shap = pd.concat(shap_all).reset_index()
    feats = pd.concat([frames[h].loc[shap_all[i].index, feature_columns(MAIN_FW)].assign(horizon=h)
                       for i, h in enumerate(config.HORIZONS)]).reset_index()
    assert len(preds) == len(shap) == len(feats)

    importance = {}
    for h in config.HORIZONS:
        s = shap[shap["horizon"] == h].drop(columns=["index", "horizon", "base_value"])
        importance[f"h{h}"] = s.abs().mean().sort_values(ascending=False).round(4).to_dict()

    art = config.ARTIFACTS_DIR
    (art / "models").mkdir(parents=True, exist_ok=True)
    preds.drop(columns=["index"]).to_parquet(art / "predictions.parquet", index=False, compression="zstd")
    shap.drop(columns=["index"]).to_parquet(art / "shap.parquet", index=False, compression="zstd")
    feats.drop(columns=["index"]).to_parquet(art / "features.parquet", index=False, compression="zstd")
    (art / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (art / "importance.json").write_text(json.dumps(importance, indent=2, ensure_ascii=False), encoding="utf-8")

    # Modelos finales con todo el historial (para pronósticos operativos futuros).
    for h in config.HORIZONS:
        hm = models.fit_horizon(frames[h], h, params, stations, future_weather=MAIN_FW)
        for q, m in hm.quantile.items():
            m.booster_.save_model(str(art / "models" / f"h{h}_q{int(q * 100):02d}.txt"))
        hm.classifier.booster_.save_model(str(art / "models" / f"h{h}_naranja.txt"))
    (art / "models" / "meta.json").write_text(json.dumps({
        "features": feature_columns(MAIN_FW), "usa_meteo_futura": MAIN_FW, "stations": stations, "params": params,
        "umbral_alerta": thresholds, "desplazamiento_conformal": {h: offsets[h][max(offsets[h])] for h in offsets},
        "entrenado_hasta": str(daily["date"].max().date()),
    }, indent=2), encoding="utf-8")

    print_table(metrics)
    print(f"Listo en {time.time() - t0:.0f} s")


def summarize_tuning(oos: pd.DataFrame) -> dict:
    out = {}
    for h, g in oos.groupby("horizon"):
        sub = g[g["target_date"].dt.year.isin(models.TUNING_YEARS)]
        y = sub["y"].to_numpy()
        ref = float(np.mean(np.abs(sub["base_persistencia"] - y)))
        out[f"h{h}"] = {"n": int(len(sub)), "LightGBM (BRUMA)": models.point_metrics(y, sub["q50"].to_numpy(), ref),
                        ALT_LABEL: models.point_metrics(y, sub["q50_variante"].to_numpy(), ref),
                        "Persistencia": models.point_metrics(y, sub["base_persistencia"].to_numpy(), ref)}
    return out


def print_table(metrics: dict) -> None:
    print("\nEvaluación 2024-2025 (fuera de muestra)")
    for h, per in metrics["evaluacion"].items():
        m = per["2024-2025"]
        print(f"\n{h}  n={m['n']}")
        for name, v in m["puntual"].items():
            print(f"  {name:<36} MAE {v['MAE']:.2f}  RMSE {v['RMSE']:.2f}  R2 {v['R2']:.3f}  "
                  f"skill {v.get('skill_vs_persistencia', 0):+.1%}")
        ic, cc = m["intervalo_crudo"], m["intervalo_conformal"]
        print(f"  banda 80 %: cruda {ic['cobertura_80']:.1%} (ancho {ic['ancho_medio']:.1f}) · "
              f"conformal {cc['cobertura_80']:.1%} (ancho {cc['ancho_medio']:.1f})")
        a = m["alerta_naranja"]
        print(f"  naranja: n={a['n_dias_naranja']} AUC {a.get('roc_auc', float('nan')):.3f} "
              f"PR-AUC {a.get('pr_auc', float('nan')):.3f} BSS {a['brier_skill_vs_climatologia']:+.2f} "
              f"recall {a['recall']} precision {a['precision']}")


if __name__ == "__main__":
    main()
