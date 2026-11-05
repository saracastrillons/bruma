import numpy as np
import pandas as pd
import pytest

from bruma.features import build_panel, feature_columns, make_horizon_frame


def synthetic(days=80, stations=(1, 2), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    rows = []
    for s in stations:
        for d in dates:
            v = 20 + 5 * np.sin(d.dayofyear / 9) + rng.normal(0, 1)
            rows.append({"station": s, "date": d, "pm25": v, "pm25_evening": v + 1, "pm25_max": v + 5, "n_hours": 24})
    daily = pd.DataFrame(rows)
    weather = pd.DataFrame({"date": dates})
    for c in ["temp_mean", "temp_range", "rh_mean", "precip_sum", "wind_mean", "u_mean", "v_mean",
              "rad_sum", "cloud_mean", "blh_day", "blh_night", "pressure_mean"]:
        weather[c] = rng.normal(10, 2, len(dates))
    return daily, weather


@pytest.mark.parametrize("h", [1, 2, 3])
def test_target_is_mean_of_day_d_plus_h(h):
    daily, weather = synthetic()
    df = make_horizon_frame(build_panel(daily, weather), weather, h, future_weather=False)
    row = df[(df.station == 1) & (df.date == "2023-02-01")].iloc[0]
    expected = daily[(daily.station == 1) & (daily.date == pd.Timestamp("2023-02-01") + pd.Timedelta(days=h))].pm25.item()
    assert row.y == pytest.approx(expected)
    assert row.pm_d0 == pytest.approx(daily[(daily.station == 1) & (daily.date == "2023-02-01")].pm25.item())


@pytest.mark.parametrize("h", [1, 3])
def test_no_leakage_features_ignore_the_future(h):
    """Cambiar todo lo que ocurre después de D no debe cambiar ninguna variable de la fila D."""
    daily, weather = synthetic()
    cut = pd.Timestamp("2023-02-15")
    base = make_horizon_frame(build_panel(daily, weather), weather, h, future_weather=False)

    daily2, weather2 = daily.copy(), weather.copy()
    daily2.loc[daily2.date > cut, ["pm25", "pm25_evening"]] = 999.0
    num = weather2.columns.drop("date")
    weather2.loc[weather2.date > cut, num] = -999.0
    alt = make_horizon_frame(build_panel(daily2, weather2), weather2, h, future_weather=False)

    cols = [c for c in feature_columns(False) if c != "station"]
    a = base[base.date == cut].set_index("station")[cols]
    b = alt[alt.date == cut].set_index("station")[cols]
    pd.testing.assert_frame_equal(a, b)


def test_same_weekday_last_week_baseline_alignment():
    daily, weather = synthetic()
    h = 2
    df = make_horizon_frame(build_panel(daily, weather), weather, h, future_weather=False)
    row = df[(df.station == 2) & (df.date == "2023-02-10")].iloc[0]
    target = pd.Timestamp("2023-02-10") + pd.Timedelta(days=h)
    expected = daily[(daily.station == 2) & (daily.date == target - pd.Timedelta(days=7))].pm25.item()
    assert row.pm_same_dow_last_week == pytest.approx(expected)


def test_future_weather_columns_only_when_requested():
    daily, weather = synthetic()
    panel = build_panel(daily, weather)
    assert "blh_day_t" not in make_horizon_frame(panel, weather, 1, future_weather=False)
    assert "blh_day_t" in make_horizon_frame(panel, weather, 1, future_weather=True)
    assert len(feature_columns(True)) > len(feature_columns(False))
