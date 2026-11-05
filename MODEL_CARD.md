# Model card · BRUMA v1.0

Pronóstico del promedio de 24 h de PM2.5 a 1, 2 y 3 días para 20 estaciones del Valle de Aburrá (Antioquia, Colombia), con banda de incertidumbre del 80 %, probabilidad de superar el nivel naranja del ICA y explicación SHAP.

Autora: **Sara Castrillón Sánchez** ([saracastrillons](https://github.com/saracastrillons)) · Noviembre de 2026 · Proyecto de portafolio, no es un sistema oficial de alertas.

---

## 1. Uso previsto

| | |
|---|---|
| **Para qué sirve** | Demostrar un flujo de ciencia de datos reproducible y honesto con datos reales: anticipar si los próximos días tendrán mala calidad del aire y explicar qué señales usa el modelo. |
| **Para quién** | Personas que evalúan el proyecto (reclutadores, docentes) y curiosos del tema en Medellín. |
| **Para qué NO sirve** | No reemplaza los pronósticos ni las alertas oficiales de SIATA / Área Metropolitana. No debe usarse para decisiones de salud, de movilidad (pico y placa ambiental) ni regulatorias. No identifica causas de la contaminación. |

## 2. Datos

| Fuente | Contenido | Uso |
|---|---|---|
| [datos.gov.co · Calidad del Aire en Colombia](https://www.datos.gov.co/resource/g4t8-zkc3) (IDEAM/SISAIRE) | PM2.5 horario de la red AMVA (operada por SIATA), 2020-2025 | Objetivo y variables rezagadas |
| [SIATA · EntregaData1](https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_pm25.json) | PM2.5 horario jul-2023 a jul-2024 | Solo para verificar la fuente principal |
| [Open-Meteo · ERA5](https://open-meteo.com/en/docs/historical-weather-api) | Meteorología horaria en un punto del valle (6.25 N, -75.58 W) | Variables meteorológicas |

**Hallazgos de calidad de datos (ver `artifacts/data_quality.json` y `notebooks/01_eda.ipynb`):**

- **Cambio de escala.** 14 de las 20 estaciones publican en datos.gov.co, entre 2020 y 2023, valores que no son µg/m³ (medianas mensuales de 100 a 2.000). En el mismo periodo, el archivo propio de SIATA tiene valores normales para esas estaciones. Desde enero de 2024 ambas fuentes coinciden exactamente (correlación 1,0; error medio 0,004 µg/m³). Una regla documentada descarta esos periodos: **409.595 horas (44 % de las filas)**.
- **Hora de referencia.** SIATA marca el *fin* de la hora y datos.gov.co el *inicio*. Con un desfase de 1 h coinciden 1:1.
- Se eliminaron además 6 horas de sensor pegado y 7 picos aislados. Un promedio diario solo se calcula con al menos 18 horas válidas (criterio del 75 %).
- **Cobertura resultante:** 6 estaciones con serie 2020-2025 (más La Ye hasta 2022 y parte de 2024) y 13 estaciones solo desde 2024. Total: 21.934 estación-días válidos.
- **Meteorología faltante:** Open-Meteo no publica la altura de la capa de mezcla ERA5 entre el 31-dic-2023 y el 30-jun-2024 (justo el episodio de marzo de 2024). Se deja como faltante; LightGBM la trata de forma nativa.
- ERA5 tiene ~25 km de resolución: **un solo punto no representa la meteorología de un valle de ~10 km de ancho**.

## 3. Modelo

- **Unidad de pronóstico:** se emite al cierre del día D con información hasta las 23:00 de D y se pronostica el promedio de 24 h de D+1, D+2 y D+3 (un modelo directo por horizonte).
- **Algoritmo:** LightGBM, un modelo global para todas las estaciones (la estación entra como variable categórica).
  - 3 regresores de cuantiles (0,1 · 0,5 · 0,9) → mediana y banda.
  - 1 clasificador binario → P(promedio 24 h > 37 µg/m³).
- **Variables (33):** PM2.5 de hoy, de los días anteriores, de la noche (18-23 h) y promedios de 3, 7 y 28 días; nivel medio del valle; meteorología del día de emisión (altura de mezcla diurna y nocturna, lluvia, humedad, viento, radiación, nubosidad, presión, lluvia de 3 días); calendario del día objetivo (día de la semana, época del año, festivos colombianos, cuarentena de 2020).
- **Hiperparámetros:** `num_leaves=15, min_child_samples=30, learning_rate=0.03, n_estimators=500`, elegidos entre 4 configuraciones por MAE en los pliegues de ajuste.
- **Incertidumbre:** los cuantiles crudos se recalibran con predicción conformal (CQR, Romano et al. 2019). El ajuste de cada año se estima solo con residuos fuera de muestra de años anteriores.

## 4. Evaluación

**Protocolo (origen móvil por años, sin fuga):**

| Pliegue | Entrena con objetivos | Prueba | Se usa para |
|---|---|---|---|
| 2022 | < 2022 | 2022 | elegir hiperparámetros y umbral de alerta |
| 2023 | < 2023 | 2023 | elegir hiperparámetros y umbral de alerta |
| 2024 | < 2024 | 2024 | **evaluación final** |
| 2025 | < 2025 | 2025 | **evaluación final** |

Una prueba automática (`tests/test_features.py`) verifica que alterar cualquier dato posterior a D no cambia ninguna variable de la fila D.

**Resultados en 2024-2025 (13.187 pronósticos estación-día por horizonte):**

| Modelo | MAE 24 h | MAE 48 h | MAE 72 h | R² 24 h |
|---|---|---|---|---|
| **BRUMA (LightGBM)** | **2,69** | **3,19** | **3,37** | **0,76** |
| Regresión Ridge (mismas variables) | 2,70 | 3,32 | 3,54 | 0,75 |
| Persistencia (mañana = hoy) | 3,23 | 4,01 | 4,34 | 0,64 |
| Estacional semanal | 4,37 | 4,42 | 4,43 | 0,27 |
| Climatología estación-mes | 4,90 | 4,90 | 4,90 | 0,19 |

Mejora frente a la persistencia: **−17 %, −20 % y −22 %** de MAE. A 24 h la regresión lineal casi empata: la mayor parte de la señal es la persistencia de la contaminación.

**Incertidumbre (banda nominal del 80 %):**

| | 24 h | 48 h | 72 h |
|---|---|---|---|
| Cobertura con cuantiles crudos | 67,9 % | 67,3 % | 68,3 % |
| Cobertura con conformal | **82,2 %** | **83,4 %** | **84,2 %** |
| Ancho medio con conformal (µg/m³) | 9,0 | 10,7 | 11,6 |

**Probabilidad de nivel naranja** (218 estación-días en naranja, tasa base 1,7 %):

| | 24 h | 48 h | 72 h |
|---|---|---|---|
| ROC-AUC | 0,978 | 0,940 | 0,938 |
| PR-AUC | 0,472 | 0,190 | 0,212 |
| Brier skill vs climatología mensual | +0,23 | 0,00 | +0,06 |
| Umbral de aviso (elegido en 2022-2023) | 0,325 | 0,075 | 0,075 |
| Recall / precisión con ese umbral | 31 % / 50 % | 23 % / 25 % | 26 % / 26 % |

**Ablación: meteorología del día objetivo.** Añadir la meteorología ERA5 del día pronosticado (equivalente a un pronóstico del tiempo *perfecto*) mejora el MAE ~4-5 % en 2022-2023 pero no en 2024-2025 (+1 % a 24 h, peor a 48-72 h). Como esa información no existe al emitir y su ganancia no es estable, el modelo operativo no la usa.

## 5. Limitaciones y sesgos conocidos

1. **Subestima los episodios.** En feb-abr el MAE sube a 3,3 / 4,4 / 4,8 µg/m³ y la cobertura de la banda baja a ~77 % (24 h). Un modelo mediano tiende a regresar hacia la media justo cuando más importa.
2. **La alerta pierde la mayoría de los días naranja.** A 24 h, con el umbral elegido, detecta 31 % de los eventos; a 48-72 h la utilidad es baja. El umbral se eligió con solo ~26 eventos de 2022-2023.
3. **Los eventos de evaluación están concentrados:** 210 de los 218 días naranja son de 2024 (sobre todo marzo). 2025 tuvo solo 8. Las métricas de alerta tienen mucha varianza.
4. **Estaciones nuevas.** 13 estaciones solo tienen datos desde 2024: el modelo de 2024 las pronosticó sin haberlas visto nunca (aprende de sus rezagos, no de su historia).
5. **Datos de la propia fuente.** Se confía en la validación de IDEAM/SIATA. Los valores marcados como dudosos por SIATA no vienen en datos.gov.co.
6. **Meteorología gruesa** (un punto ERA5) y con un hueco de 6 meses en la altura de mezcla.
7. **No es causal.** SHAP describe cómo el modelo combina sus variables. Que "PM2.5 de esta noche" tenga el mayor aporte no significa que lo cause; refleja persistencia.
8. **Modo repetición.** El tablero reproduce 2024-2025. No hay ingesta en vivo: el dataset público llega hasta el 31-dic-2025 y el servicio en vivo de SIATA no respondió durante el desarrollo.

## 6. Explicabilidad

Valores SHAP exactos de TreeSHAP (`pred_contrib` de LightGBM) del modelo mediano. Se suman al valor base y dan la predicción (lo verifica una prueba automática). Importancia global (media |SHAP|, 24 h): PM2.5 de la noche anterior > PM2.5 del día > promedio de 28 días > promedio de 7 días > día de la semana. A 48-72 h dominan los promedios de 7 y 28 días y el día de la semana.

## 7. Boletín con IA

Opcional. Gemini redacta el boletín **solo** a partir de un JSON con salidas del modelo (pronósticos, bandas, probabilidades y SHAP agrupado). Si el texto menciona un número que no está en ese JSON, se descarta y se usa la plantilla determinista. Sin `GEMINI_API_KEY` siempre se usa la plantilla.

## 8. Reproducibilidad

```
py scripts/download_data.py   # descarga y cachea (data/raw)
py scripts/build_dataset.py   # limpieza -> data/processed + artifacts/data_quality.json
py scripts/train.py           # validación, métricas, SHAP -> artifacts/
```

Semilla fija (42). El entrenamiento completo tarda ~4 minutos en un portátil.
