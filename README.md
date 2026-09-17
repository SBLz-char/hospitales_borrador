---
title: Predicción Ocupación Hospitalaria
emoji: 🏥
colorFrom: teal
colorTo: blue
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
---

# Predicción de Ocupación Hospitalaria

App de predicción del índice ocupacional de camas para hospitales públicos chilenos
(Red REM 20), parte de la tesis de Juanfra Mathias.

Hay **dos interfaces** sobre la misma lógica, para poder desplegar en Streamlit
Community Cloud (gratis) sin depender de la suscripción PRO que Hugging Face empezó a
pedir para Spaces con Gradio/Docker en cuentas nuevas (agosto 2026).

## Archivos

- `streamlit_app.py` — interfaz de **Streamlit** (punto de entrada recomendado hoy —
  ver "Cómo desplegar en Streamlit Community Cloud" más abajo).
- `app.py` — interfaz de **Gradio** (para Hugging Face Spaces, si más adelante se paga
  PRO o se recupera un Space anterior).
- `logica.py` — toda la lógica de negocio compartida entre las dos interfaces (carga
  del modelo, combinar lo subido con lo conocido, armar la predicción, el gráfico).
  Ninguna de las dos interfaces duplica esta lógica.
- `pipeline.py` — construcción de features (rezagos, complejidad, encodings).
- `inferencia.py` — predicción recursiva de 1 a 3 meses, fuzzy matching de hospitales,
  semáforo.
- `carga_datos.py` — lectura y validación del CSV que sube el usuario.
- `modelo_ocupacion_hospitalaria_v3.pkl` — modelo LightGBM entrenado (Avance 05).
- `dataset_referencia.parquet` — historial de los 208 hospitales de la Red REM 20.
- `requirements.txt` — dependencias (de ambas interfaces).

## Horizonte de predicción

Los horizontes son **1, 2 o 3 meses**, contados desde el último mes con dato real de ese
hospital y área (no desde la fecha de hoy). Se definen en `logica.HORIZONTES` y el tope
está en `inferencia.HORIZONTE_MAXIMO`: pedir un mes más lejano levanta
`HorizonteFueraDeRangoError` en vez de proyectar.

El motivo es que cada mes proyectado se reinyecta como `LAG1` del siguiente. A 3 meses
solo `ROLL3` conserva un dato real, y desde el mes 4 los tres rezagos serían predicciones
del propio modelo. Medido sobre el período de test, el error crece con el horizonte
(MAE 7,8 a 1 mes, 9,2 a 2 y 10,1 a 3), y más allá de 3 meses no aporta información nueva.

Las dos interfaces ofrecen las mismas opciones, porque las leen de `logica.HORIZONTES`.
Si llega una etiqueta que no está en ese diccionario, la app muestra un error en vez de
predecir a otro horizonte en silencio.

## Cómo correr localmente

```
pip install -r requirements.txt
streamlit run streamlit_app.py     # interfaz de Streamlit
python app.py                      # o, alternativamente, la de Gradio
```

## Cómo desplegar en Streamlit Community Cloud (gratis)

1. Crear un repositorio en GitHub (puede ser privado o público) y subir todos estos
   archivos a la raíz del repo — con `git`, o arrastrándolos desde la web de GitHub
   ("Add file" → "Upload files").
2. Entrar a [share.streamlit.io](https://share.streamlit.io) con la cuenta de GitHub.
3. "New app" → elegir el repositorio, la rama (`main`), y como "Main file path"
   escribir `streamlit_app.py`.
4. "Deploy". Streamlit instala `requirements.txt` y corre la app — toma unos minutos
   la primera vez.
5. La URL pública queda como `tu-usuario-tu-repo.streamlit.app` (se puede personalizar
   el subdominio al desplegar).

## Cómo desplegar en Hugging Face Spaces (si más adelante aplica)

1. Crear un Space nuevo con SDK "Gradio" (requiere PRO en cuentas nuevas desde agosto
   2026 — o reutilizar un Space ya existente de antes de esa fecha).
2. Subir estos archivos a la raíz del Space (incluyendo el `.pkl` y el `.parquet`).
3. El Space instala `requirements.txt` y corre `app.py` (definido en el YAML de arriba).

## Pendientes conocidos (no bloquean esta versión de prueba)

- La predicción recursiva no muestra todavía una banda de incertidumbre creciente
  con el horizonte.
- El dropdown de área funcional no depende (todavía) del hospital elegido.
- Cuando el área o la zona de un hospital nuevo no calza con ninguna categoría
  conocida, el encoding cae al promedio general sin avisarlo explícitamente en la
  interfaz.
- Soporte de Excel (además de CSV) queda para una siguiente versión.
