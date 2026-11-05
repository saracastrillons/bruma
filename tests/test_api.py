"""Pruebas de extremo a extremo sobre los artefactos versionados en el repo."""
import json

import pytest
from fastapi.testclient import TestClient

from bruma import config
from bruma.api.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def test_healthz(client):
    r = client.get("/healthz").json()
    assert r["ok"] and r["estaciones"] == 20


def test_meta(client):
    m = client.get("/api/meta").json()
    assert m["fechas"]["min"] < m["fechas"]["sugerida"] <= m["fechas"]["max"]
    assert set(m["metricas"]) == {"h1", "h2", "h3"}


def test_forecast_shape_and_shap_additivity(client):
    f = client.get("/api/forecast", params={"station": 29509, "date": "2024-03-04"}).json()
    assert f["fecha_emision"] == "2024-03-04"
    assert [h["h"] for h in f["horizontes"]] == [1, 2, 3]
    for h in f["horizontes"]:
        assert h["q10"] <= h["q50"] <= h["q90"]
        ex = h["explicacion"]
        total = ex["base"] + sum(i["aporte"] for i in ex["impulsores"]) + ex["otros"]
        assert total == pytest.approx(ex["prediccion"], abs=0.05)
        assert ex["prediccion"] == pytest.approx(h["q50"], abs=0.1)
    assert any(d["futuro"] for d in f["historia"])


def test_errors(client):
    assert client.get("/api/forecast", params={"station": 1, "date": "2024-03-04"}).status_code == 404
    assert client.get("/api/valley", params={"date": "2019-01-01"}).status_code == 404
    assert client.get("/api/valley", params={"h": 7}).status_code == 422


def test_valley_and_bulletin(client):
    v = client.get("/api/valley", params={"date": "2024-03-04", "h": 1}).json()
    assert len(v["estaciones"]) >= 15
    assert all(0 <= s["p_naranja"] <= 1 for s in v["estaciones"])
    b = client.get("/api/bulletin", params={"date": "2024-03-04"}).json()
    assert b["fuente"] == "plantilla" and "µg/m³" in b["texto"]


def test_committed_metrics_beat_persistence_and_bands_are_calibrated():
    m = json.loads((config.ARTIFACTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    for h, per in m["evaluacion"].items():
        e = per["2024-2025"]
        assert e["puntual"]["LightGBM (BRUMA)"]["skill_vs_persistencia"] > 0.1, h
        assert 0.75 <= e["intervalo_conformal"]["cobertura_80"] <= 0.9, h
