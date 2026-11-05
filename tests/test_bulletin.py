from bruma import bulletin

FACTS = bulletin.build_facts("2024-03-04", [{
    "h": 1, "etiqueta": "Mañana (24 h)", "fecha": "2024-03-05", "promedio_valle": 36.4,
    "min": 33.1, "estacion_min": "Tanques La Ye", "max": 41.4, "estacion_max": "Tráfico Centro",
    "estaciones_en_alerta": ["Tráfico Centro", "Casa Justicia"], "prob_max": 0.98,
    "impulsores": {"Contaminación acumulada en la estación": 12.6, "Nivel de contaminación del valle": 3.7},
}])


def test_template_is_grounded_and_mentions_key_numbers():
    text = bulletin.deterministic_bulletin(FACTS)
    assert "36 µg/m³" in text and "98 %" in text and "Tráfico Centro" in text
    assert bulletin.grounded(text, FACTS)


def test_grounding_rejects_invented_numbers():
    assert bulletin.grounded("Se esperan 36 µg/m³ y 98 % de probabilidad.", FACTS)
    assert not bulletin.grounded("Se esperan 52 µg/m³ por los incendios.", FACTS)


def test_falls_back_to_template_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out = bulletin.make_bulletin(FACTS)
    assert out["fuente"] == "plantilla"


def test_ungrounded_llm_output_is_discarded(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")

    class FakeModels:
        def generate_content(self, **_):
            return type("R", (), {"text": "Mañana habrá 80 µg/m³ y 5000 muertes."})()

    class FakeClient:
        def __init__(self, **_):
            self.models = FakeModels()

    import google.genai as genai
    monkeypatch.setattr(genai, "Client", FakeClient)
    assert bulletin.make_bulletin(FACTS)["fuente"] == "plantilla"


def test_grouping_of_features():
    assert bulletin.group_of("pm_mean7").startswith("Contaminación")
    assert bulletin.group_of("blh_day_d0").startswith("Condiciones")
    assert bulletin.group_of("dow").startswith("Calendario")
    g = bulletin.grouped_contributions({"pm_d0": 2.0, "pm_d1": 1.0, "dow": -0.5, "base_value": 18})
    assert list(g.items()) == [("Contaminación acumulada en la estación", 3.0),
                               ("Calendario (día de la semana, época del año, festivos)", -0.5)]
