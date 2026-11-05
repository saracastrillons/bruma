"""Boletín diario en lenguaje natural, anclado a las salidas del modelo.

1. ``build_facts`` reúne SOLO números producidos por el modelo (pronósticos,
   bandas, probabilidades y contribuciones SHAP).
2. ``deterministic_bulletin`` redacta el boletín con una plantilla: siempre
   funciona y es reproducible.
3. ``gemini_bulletin`` (opcional, requiere ``GEMINI_API_KEY``) pide a Gemini que
   redacte con esos hechos. La respuesta se valida: si menciona un número que no
   está en los hechos, se descarta y se usa la plantilla.
"""
from __future__ import annotations

import json
import os
import re

from . import config

GROUPS = {
    "Contaminación acumulada en la estación": ("pm_",),
    "Nivel de contaminación del valle": ("valley_",),
    "Calendario (día de la semana, época del año, festivos)": (
        "dow", "month", "doy_", "is_holiday", "is_weekend", "lockdown"),
    "Condiciones meteorológicas recientes": (
        "temp_", "rh_", "precip", "wind_", "u_mean", "v_mean", "rad_", "cloud_", "blh_", "pressure_", "rain_"),
    "Perfil típico de la estación": ("station",),
}


def group_of(feature: str) -> str:
    for name, prefixes in GROUPS.items():
        if any(feature == p or feature.startswith(p) for p in prefixes):
            return name
    return "Otros"


def grouped_contributions(shap_row: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for f, v in shap_row.items():
        if f == "base_value":
            continue
        g = group_of(f)
        out[g] = out.get(g, 0.0) + float(v)
    return dict(sorted(out.items(), key=lambda kv: -abs(kv[1])))


def build_facts(issue_date: str, horizons: list[dict]) -> dict:
    """``horizons``: [{h, fecha, promedio_valle, min, max, estacion_min, estacion_max,
    estaciones_en_alerta, prob_max, impulsores: {grupo: µg/m³}}] ya redondeados."""
    return {
        "fecha_emision": issue_date,
        "umbral_naranja_ug_m3": config.ICA_NARANJA,
        "pronosticos": horizons,
        "aviso": "Pronóstico estadístico fuera de muestra; las contribuciones SHAP describen "
                 "asociaciones aprendidas por el modelo, no causas.",
    }


def _fmt(x: float) -> str:
    return f"{x:.0f}"


def deterministic_bulletin(facts: dict) -> str:
    lines = []
    for p in facts["pronosticos"]:
        cat = config.ica_category(p["promedio_valle"])["nombre"].lower()
        head = (f"**{p['etiqueta']} ({p['fecha']}):** PM2.5 promedio esperado en el valle de "
                f"{_fmt(p['promedio_valle'])} µg/m³ ({cat}), entre {_fmt(p['min'])} µg/m³ en "
                f"{p['estacion_min']} y {_fmt(p['max'])} µg/m³ en {p['estacion_max']}.")
        if p["estaciones_en_alerta"]:
            names = p["estaciones_en_alerta"]
            listed = ", ".join(names[:3]) + (f" y {len(names) - 3} más" if len(names) > 3 else "")
            alert = (f" Riesgo de nivel naranja en {len(names)} estación(es): {listed} "
                     f"(probabilidad máxima {p['prob_max'] * 100:.0f} %).")
        else:
            alert = f" Sin estaciones sobre el umbral de alerta (probabilidad máxima {p['prob_max'] * 100:.0f} %)."
        drivers = list(p["impulsores"].items())[:2]
        why = " Lo que más mueve el pronóstico: " + "; ".join(
            f"{g.lower()} ({'+' if v >= 0 else '−'}{abs(v):.1f} µg/m³)" for g, v in drivers) + "."
        lines.append(head + alert + why)
    lines.append("_" + facts["aviso"] + "_")
    return "\n\n".join(lines)


PROMPT = """Eres el redactor del boletín de calidad del aire del Valle de Aburrá.
Escribe en español, en 3 párrafos cortos (uno por horizonte), tono claro y sereno.
REGLAS ESTRICTAS:
- Usa ÚNICAMENTE los datos del JSON. No inventes cifras, causas, fuentes ni recomendaciones médicas.
- Redondea a enteros los µg/m³ y las probabilidades en %, tal como vienen o a la unidad.
- Los "impulsores" son contribuciones SHAP del modelo: descríbelos como factores asociados
  al pronóstico, nunca como causas comprobadas.
- Termina con una frase que diga que es un pronóstico estadístico.
JSON:
{facts}"""


def _numbers(text: str) -> set[float]:
    return {float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", text)}


def grounded(text: str, facts: dict) -> bool:
    """Todo número del texto debe poder salir de los hechos (con redondeo)."""
    allowed: set[float] = set()

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            for v in (x, x * 100):
                allowed.update({round(v), round(v, 1), float(int(v))})
        elif isinstance(x, str):
            allowed.update(_numbers(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
            for k in x:
                allowed.update(_numbers(str(k)))
        elif isinstance(x, list):
            allowed.add(len(x))
            for v in x:
                walk(v)
    walk(facts)
    allowed.update({1, 2, 3, 24, 48, 72, 2.5, 80})  # horizontes, "PM2.5", banda del 80 %
    return all(any(abs(n - a) < 0.051 for a in allowed) for n in _numbers(text))


def gemini_bulletin(facts: dict) -> str | None:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=PROMPT.format(facts=json.dumps(facts, ensure_ascii=False, indent=1)),
        )
        text = (resp.text or "").strip()
    except Exception:  # red, cuota, clave inválida… el tablero nunca debe caerse por esto
        return None
    return text if text and grounded(text, facts) else None


def make_bulletin(facts: dict) -> dict:
    text = gemini_bulletin(facts)
    if text:
        return {"texto": text, "fuente": "gemini", "hechos": facts}
    return {"texto": deterministic_bulletin(facts), "fuente": "plantilla", "hechos": facts}
