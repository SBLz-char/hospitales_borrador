# Versión 1 — Explicación con IA y distancias entre hospitales

Responde las dos sugerencias del profesor guía, más el bug de Hanga Roa que apareció
después del último merge.

## Archivos

| Archivo | Estado | Qué hace |
|---|---|---|
| `geo.py` | **nuevo** | Región, comuna y distancia entre hospitales |
| `explicacion.py` | **nuevo** | El párrafo en lenguaje simple bajo el gráfico |
| `hospitales_geo.parquet` | **nuevo** | Tabla de apoyo: 192 hospitales con coordenadas (20 KB) |
| `logica.py` | modificado | Camas físicas, filtro por región, orden por cercanía |
| `streamlit_app.py` | modificado | Muestra camas, el cuadro de explicación y las distancias |
| `requirements.txt` | modificado | Agrega `anthropic>=1,<2` |

`app.py`, `carga_datos.py`, `inferencia.py`, `pipeline.py`, el `.pkl` y
`dataset_referencia.parquet` **no cambian**. Se suben estos seis con su nombre tal cual,
a la raíz del repositorio.

> `app.py` (la versión Gradio) sigue funcionando sin tocarla: `predecir_valor()` agrega
> claves nuevas al diccionario que devuelve, pero no quita ninguna de las que usa.

## 1. El dataset de distancias

**Fuente:** "Establecimientos de Salud vigentes", DEIS / Ministerio de Salud, publicada en
el Portal de Datos Abiertos del Estado bajo licencia Creative Commons CCZero.
<https://datos.gob.cl/dataset/establecimientos-de-salud-vigentes>

El archivo descargado (`establecimientos_20260929.csv`, 5.743 establecimientos, 2,4 MB)
y el script que construye la tabla están en la carpeta `distancias/`, fuera del
repositorio. Solo viaja al repo el resultado: `hospitales_geo.parquet`, 192 filas.

**El dataset del REM no se modifica.** `dataset_referencia.parquet` queda exactamente
igual; `geo.py` cruza su propia tabla por `CODIGO_ESTABLECIMIENTO` solo en el momento de
recomendar, igual que una tabla temporal. Si el REM se actualiza, esta tabla sigue
sirviendo sin cambios.

**Verificación del cruce** (la imprime `construir_hospitales_geo.py` y se detiene si algo
falla):

- 192 de 192 hospitales encontrados, 0 sin ficha
- 192 con coordenadas, 0 nulas, 0 fuera de Chile
- los 192 marcados "Vigente en Operación Habitual" por el DEIS
- 0 hospitales compartiendo coordenada exacta
- distancias de control: Instituto Asenjo a 192 m del Hospital Del Salvador, Ex Militar a
  2,7 km, Luis Tisné a 8,3 km — coherentes con el mapa

Se revisaron además los 192 pares de nombres REM ↔ DEIS. Cinco difieren por cambio de
nombre oficial, y los cinco son el mismo establecimiento en la misma comuna:

| Código | Nombre en el REM | Nombre en el DEIS |
|---|---|---|
| 125100 | Hospital Regional (Coihaique) | Hospital Regional de Coyhaique |
| 122100 | Hospital Clínico Regional (Valdivia) | Hospital Base Valdivia |
| 107101 | Hospital San Martín (Quillota) | Hospital Biprovincial Quillota Petorca |
| 122201 | Hospital Padre Bernabé de Lucerna (Panguipulli) | Hospital de Panguipulli |
| 115100 | Hospital Regional de Rancagua | Hospital Dr. Franco Ravera Zunino |

### Por qué hacía falta: el bug de Hanga Roa

De los 29 servicios de salud, **uno solo abarca dos regiones**: el Metropolitano Oriente,
porque el Hospital Hanga Roa depende de él pero está en Isla de Pascua, región de
Valparaíso, a **3.768 km** del Hospital Del Salvador. Como el motor buscaba alternativas
"del mismo servicio de salud", lo ofrecía como destino de traslado.

Ahora el filtro es *mismo servicio **y** misma región*, con la región puesta por el
Ministerio y no por una lista escrita a mano. Se verificó recorriendo todos los hospitales
de la Región Metropolitana que proyectan sobre 85%: Hanga Roa aparece **0 veces**.

### Orden de las alternativas

Tres criterios, en este orden:

1. los que tienen la **misma área** consultada van primero (criterio clínico);
2. dentro de cada grupo, el **más cercano**;
3. a igual distancia, el más desocupado.

La distancia es en línea recta, no por carretera: siempre es menor o igual a la real, y
sirve para ordenar, no para estimar un tiempo de viaje. Calcular rutas reales es posible
—son 701 pares dentro de los servicios de salud, se precalculan una vez— pero queda
pendiente de decisión.

## 2. La explicación bajo el gráfico

Regla que ordena todo el diseño:

> **Los números los calcula Python. La IA solo los redacta.**

Hay dos caminos y la app siempre tiene uno:

1. **Plantilla determinista** — sin red, sin clave, sin costo. Siempre funciona.
2. **Texto con IA** — solo si hay clave configurada, la API responde, y el texto pasa la
   verificación de números.

`explicar()` intenta (2) y cae a (1) ante cualquier problema. Sin clave, sin internet, con
la API caída o dentro de un año cuando nadie recuerde renovar nada, **la sección sigue
mostrando el párrafo determinista**. Por eso el proyecto se puede presentar en cualquier
momento sin depender de un servicio externo.

### La verificación de números

Un modelo de lenguaje puede inventar una cifra, y en salud eso no es aceptable.
`_verificar_numeros()` extrae **todos** los números del texto generado y los compara contra
los que se le entregaron (con tolerancia de 0,6 para absorber el redondeo de "unas 34
camas" cuando el dato es 33,6). Si aparece aunque sea uno que no calza, el texto completo
se descarta y se usa la plantilla. No se corrige ni se reintenta: se descarta.

### Los 9 escenarios de falla probados

Cada uno termina en un texto utilizable, ninguno llega al usuario como error:

| Escenario | Resultado |
|---|---|
| Texto correcto | se usa el de la IA |
| Menciona un número inventado | se descarta, va la plantilla |
| Respuesta vacía | plantilla |
| Respuesta cortada por el límite | plantilla |
| El modelo declina responder | plantilla |
| Clave de API inválida | plantilla |
| Sin conexión a internet | plantilla |
| La API responde error 500 | plantilla |
| Librería `anthropic` antigua | plantilla |
| Sin clave configurada | plantilla, sin intentar llamar |

Además hay caché en memoria: tres consultas iguales hacen **una** llamada a la API.

### Configurar la clave (opcional)

En Streamlit Community Cloud: **Manage app → Settings → Secrets**, y pegar:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

En local, exportar la variable de entorno `ANTHROPIC_API_KEY`. La clave nunca se escribe en
el código ni se sube al repositorio. Sin clave, la app funciona igual con la plantilla.

El modelo es `claude-opus-5-5`, en la constante `MODELO_IA` de `explicacion.py`. Si el costo
llega a importar, `claude-sonnet-5-5` o `claude-haiku-4-5` sirven igual para esta tarea
—es redacción corta a partir de cifras ya resueltas— y son más baratos. Cambiar solo esa
constante: la verificación de números no depende del modelo.

## 3. Las camas físicas

`logica.camas_en_numeros()` traduce el índice a camas usando `PROMEDIO_CAMAS_DISPONIBLE`,
que ya venía en el dataset. Dos cosas que el texto dice explícitamente:

- El índice del REM son **días-cama ocupados sobre días-cama disponibles durante el mes**,
  no una foto de un instante. Por eso se habla de "promedio del mes" y no de "quedan N
  camas ahora".
- Se asume la misma dotación del último mes conocido: el modelo predice el índice, no
  cuántas camas va a habilitar el hospital.

Las camas ocupadas y libres se redondean de forma que **siempre sumen el total**.

## Qué se probó

- `probar_logica.py` — capa geográfica, filtro por región, orden por cercanía, camas, y el
  barrido completo de hospitales de la RM buscando a Hanga Roa
- `probar_ia.py` — los 9 escenarios de falla, el prompt que se envía y el caché
- `probar_ui.py` — la app completa con `AppTest` (sin navegador): **0 excepciones**

Resultado del caso reportado (Hospital Del Salvador, Cuidados Intermedios Adultos,
2 meses):

```
93,4% para 08/2026
De 36 camas del área, unas 34 ocupadas y 2 libres (promedio del mes)

Sugerencia de traslado — Metropolitano Oriente · región Metropolitana de Santiago
                         · ordenadas por cercanía
  1. Instituto de Neurocirugía Dr. Alfonso Asenjo   192 m   misma área   82,4%
  2. Instituto Nacional de Enfermedades Resp. y C.  285 m                79,9%
```

## Pendiente de decisión

- **Rutas reales por carretera** en vez de línea recta (701 pares, se precalculan una vez
  con OpenRouteService; la app no haría ninguna llamada de red).
- El `.pkl` del repositorio sigue sin métricas por horizonte: el margen de error muestra
  ±7,4 puntos para los tres horizontes en vez de 7,8 / 9,2 / 10,1.
- La tabla del DEIS trae `NivelComplejidadEstabGlosa`, el nivel de complejidad real de cada
  hospital, que hoy el pipeline aproxima con un proxy de camas. Usarlo obligaría a
  reentrenar el modelo: queda como trabajo futuro, no como parte de esta versión.
