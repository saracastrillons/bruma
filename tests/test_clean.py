import numpy as np
import pandas as pd

from bruma import clean


def raw_frame(station, start, values):
    t = pd.date_range(start, periods=len(values), freq="h")
    return pd.DataFrame({"estacion_id": station, "med_fecha_inicio": t.astype(str),
                         "med_concentracion_estandar": values})


def test_scale_rule_drops_whole_bad_year_but_keeps_good_station():
    rng = np.random.default_rng(0)
    hours = 24 * 120  # ~4 meses
    good = raw_frame(1, "2022-01-01", rng.normal(20, 5, hours).clip(1))
    bad = raw_frame(2, "2022-01-01", rng.normal(1500, 300, hours).clip(1))
    report = {}
    out = clean.clean_hourly(pd.concat([good, bad]), report)
    assert out.loc[out.station == 2, "pm25"].isna().all()
    assert out.loc[out.station == 1, "pm25"].notna().mean() > 0.99
    assert report["periodo_escala_inconsistente"] == hours


def test_out_of_range_values_become_missing():
    values = 20 + 3 * np.sin(np.arange(24 * 31) / 5)
    values[[10, 400]] = [-3.0, 700.0]
    report = {}
    out = clean.clean_hourly(raw_frame(1, "2024-01-01", values), report)
    assert out["pm25"].isna().sum() == 2
    assert report["fuera_de_rango_o_no_numerico"] == 2
    assert report["periodo_escala_inconsistente"] == 0


def test_flatline_detects_stuck_sensor():
    s = pd.Series([5, 7, 9, 9, 9, 9, 9, 9, 4, 3], dtype=float)
    mask = clean._flatline_mask(s, 6)
    assert mask.tolist() == [False, False, True, True, True, True, True, True, False, False]


def test_isolated_spike_is_removed_but_sustained_rise_is_kept():
    spike = pd.Series([20, 21, 22, 300, 21, 20, 19], dtype=float)
    assert clean._spike_mask(spike).tolist() == [False, False, False, True, False, False, False]
    rise = pd.Series([20, 60, 110, 130, 125, 120, 115], dtype=float)
    assert not clean._spike_mask(rise).any()


def test_daily_mean_requires_18_valid_hours():
    t = pd.date_range("2024-03-01", periods=48, freq="h")
    values = np.full(48, 30.0)
    values[24:24 + 7] = np.nan  # segundo día con solo 17 horas válidas
    hourly = pd.DataFrame({"station": 1, "time": t, "pm25": values})
    daily = clean.daily_means(hourly)
    assert daily["pm25"].iloc[0] == 30.0
    assert np.isnan(daily["pm25"].iloc[1])
    assert daily["n_hours"].tolist() == [24, 17]


def test_daily_weather_wind_components():
    t = pd.date_range("2024-01-01", periods=24, freq="h")
    w = pd.DataFrame({"time": t, "temperature_2m": 20.0, "relative_humidity_2m": 70.0, "precipitation": 0.5,
                      "wind_speed_10m": 10.0, "wind_direction_10m": 0.0,  # viento DEL norte -> sopla hacia el sur
                      "shortwave_radiation": 100.0, "cloud_cover": 50.0, "boundary_layer_height": 800.0,
                      "surface_pressure": 850.0})
    d = clean.daily_weather(w).iloc[0]
    assert d["precip_sum"] == 12.0
    assert abs(d["u_mean"]) < 1e-9 and d["v_mean"] == -10.0
