import numpy as np
import pandas as pd
import pytest

from bruma import config, models
from bruma.features import build_panel, make_horizon_frame
from tests.test_features import synthetic


def test_split_year_has_no_temporal_overlap():
    df = pd.DataFrame({"target_date": pd.date_range("2021-06-01", "2024-06-01", freq="D")})
    train, test = models.split_year(df, 2023)
    assert train.target_date.max() < pd.Timestamp("2023-01-01")
    assert test.target_date.min() >= pd.Timestamp("2023-01-01")
    assert test.target_date.max() < pd.Timestamp("2024-01-01")


def test_conformal_offset_reaches_target_coverage():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 2, 5000)
    lo, hi = np.full_like(y, -1.0), np.full_like(y, 1.0)  # banda demasiado estrecha
    off = models.conformal_offset(y, lo, hi, alpha=0.2)
    y_new = rng.normal(0, 2, 5000)
    cover = np.mean((y_new >= lo - off) & (y_new <= hi + off))
    assert 0.78 < cover < 0.82


def test_pinball_loss():
    y = np.array([10.0, 10.0])
    assert models.pinball(y, np.array([8.0, 12.0]), 0.5) == pytest.approx(1.0)
    assert models.pinball(y, np.array([12.0, 12.0]), 0.9) == pytest.approx(0.2)


def test_best_f1_threshold_separates_perfect_classifier():
    y = np.array([10] * 90 + [50] * 10, dtype=float)
    p = np.array([0.01] * 90 + [0.6] * 10)
    t = models.best_f1_threshold(y, p)
    assert 0.01 < t <= 0.6


def test_exceedance_metrics_counts():
    y = np.array([10, 40, 50, 20], dtype=float)
    p = np.array([0.1, 0.8, 0.3, 0.7])
    m = models.exceedance_metrics(y, p, np.full(4, 0.5), threshold=0.5)
    assert m["n_dias_naranja"] == 2
    assert (m["verdaderos_positivos"], m["falsas_alarmas"], m["eventos_perdidos"]) == (1, 1, 1)


def test_fit_horizon_quantiles_are_ordered_and_shap_is_additive():
    daily, weather = synthetic(days=200, stations=(1, 2, 3))
    df = make_horizon_frame(build_panel(daily, weather), weather, 1, future_weather=False).dropna(subset=["y"])
    params = dict(num_leaves=7, min_child_samples=10, learning_rate=0.1, n_estimators=60)
    hm = models.fit_horizon(df, 1, params, [1, 2, 3], future_weather=False)
    pred = hm.predict(df)
    assert (pred.q10_raw <= pred.q50).all() and (pred.q50 <= pred.q90_raw).all()
    assert pred.p_naranja.between(0, 1).all()
    sv = hm.shap_values(df)
    raw_median = hm.quantile[0.5].predict(models.prepare_X(df, hm.cols, hm.stations))
    np.testing.assert_allclose(sv.sum(axis=1), raw_median, atol=1e-6)


def test_baselines_fall_back_when_today_is_missing():
    train = pd.DataFrame({"station": [1, 1], "month": [3, 3], "y": [20.0, 30.0]})
    test = pd.DataFrame({"station": [1, 2], "month": [3, 3], "pm_d0": [np.nan, 40.0],
                         "pm_mean7": [18.0, np.nan], "pm_same_dow_last_week": [np.nan, 33.0]})
    b = models.baseline_predictions(train, test)
    assert b["persistencia"].tolist() == [18.0, 40.0]
    assert b["climatologia"].tolist() == [25.0, 25.0]
    assert b["estacional_semanal"].tolist() == [18.0, 33.0]


def test_ica_category_boundaries():
    assert config.ica_category(12)["nombre"] == "Buena"
    assert config.ica_category(37)["nombre"] == "Aceptable"
    assert config.ica_category(37.5)["nombre"] == "Dañina a grupos sensibles"
