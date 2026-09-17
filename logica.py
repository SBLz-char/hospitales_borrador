"""
Lógica de negocio compartida entre las dos interfaces (app.py con Gradio y
streamlit_app.py con Streamlit) -- carga del modelo/dataset de referencia, y las
funciones que combinan lo subido con lo conocido y arman una predicción. Ninguna
función de acá depende de Gradio ni de Streamlit: cada interfaz llama a
predecir_valor() y decide cómo mostrar el resultado con sus propios componentes.
"""
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from carga_datos import ArchivoVacioError, ColumnasFaltantesError, cargar_csv_hospital
from inferencia import (
    HorizonteFueraDeRangoError,
    MesFueraDeRangoError,
    SinHistorialError,
    construir_serie_historica,
    evaluar_semaforo,
    predecir_recursivo,
    sugerir_coincidencias,
)
from pipeline import clasificar_complejidad

DIR_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(DIR_BASE, 'modelo_ocupacion_hospitalaria_v3.pkl')
RUTA_REFERENCIA = os.path.join(DIR_BASE, 'dataset_referencia.parquet')

HORIZONTES = {'1 mes': 1, '2 meses': 2, '3 meses': 3}

# ── Se carga una sola vez al iniciar la app, no en cada predicción (latencia mínima
#    por request). ────────────────────────────────────────────────────────────────
_artefacto = joblib.load(RUTA_MODELO)
MODELO = _artefacto['modelo']
FEATURES = _artefacto['features']
ARTEFACTOS = _artefacto['artefactos']

DF_REFERENCIA = pd.read_parquet(RUTA_REFERENCIA)


def _canon_por_codigo(df_subido=None):
    """Diccionarios {CODIGO_ESTABLECIMIENTO: nombre} y {COD_AREA_FUNCIONAL: nombre} con
    el nombre del mes más reciente en que aparece cada código.

    El REM 20 reescribe los nombres entre años (ej. 'Hospital Del Salvador (Valparaíso)'
    hasta 2022 y 'Hospital Del Salvador de Valparaíso' desde 2023): son el mismo hospital
    porque comparten CODIGO_ESTABLECIMIENTO, y la interfaz debe mostrar uno solo. Si el
    archivo subido y la referencia traen nombres distintos para el mismo mes, gana el
    archivo subido, igual que con los datos repetidos de dataframe_activo().

    No se toca GLOSA_SSS: el modelo la usa como feature (GLOSA_SSS_ENC) y solo conoce los
    nombres antiguos de los servicios de salud, así que unificarla al nombre nuevo haría
    caer el encoding al promedio general sin avisar."""
    partes = [DF_REFERENCIA.assign(_SUBIDO=0)]
    if df_subido is not None and not df_subido.empty:
        partes.append(df_subido.assign(_SUBIDO=1))
    d = pd.concat(partes, ignore_index=True) if len(partes) > 1 else partes[0]
    # los espacios sobrantes ya se limpian al leer el CSV (carga_datos), pero se repite acá
    # para que la unificación no dependa de por dónde entraron los datos
    d = d.assign(_ORDEN=d['PERIODO'] * 12 + d['MES'],
                 ESTABLECIMIENTO=d['ESTABLECIMIENTO'].astype(str).str.strip(),
                 AREA_FUNCIONAL=d['AREA_FUNCIONAL'].astype(str).str.strip())
    # sort estable + last(): el último es el mes más nuevo y, dentro del mismo mes, lo subido
    d = d.sort_values(['_ORDEN', '_SUBIDO'], kind='stable')
    return (d.groupby('CODIGO_ESTABLECIMIENTO')['ESTABLECIMIENTO'].last().to_dict(),
            d.groupby('COD_AREA_FUNCIONAL')['AREA_FUNCIONAL'].last().to_dict())


def _aplicar_canon(df, canon_hospitales, canon_areas):
    """Reescribe los nombres por el canónico de su código, dejando el original si el
    código no está en el diccionario."""
    df = df.copy()
    df['ESTABLECIMIENTO'] = df['CODIGO_ESTABLECIMIENTO'].map(canon_hospitales).fillna(df['ESTABLECIMIENTO'])
    df['AREA_FUNCIONAL'] = df['COD_AREA_FUNCIONAL'].map(canon_areas).fillna(df['AREA_FUNCIONAL'])
    return df


CANON_HOSPITALES, CANON_AREAS = _canon_por_codigo()
DF_REFERENCIA = _aplicar_canon(DF_REFERENCIA, CANON_HOSPITALES, CANON_AREAS)
HOSPITALES_REFERENCIA = sorted(DF_REFERENCIA['ESTABLECIMIENTO'].unique().tolist())
AREAS_REFERENCIA = sorted(DF_REFERENCIA['AREA_FUNCIONAL'].unique().tolist())


def dataframe_activo(df_subido):
    """El dataframe de referencia (208 hospitales) más lo que haya subido esta sesión,
    con los nombres de hospital y área unificados por código (ver _canon_por_codigo).
    df_subido debe vivir en un estado por-sesión (gr.State en Gradio, st.session_state
    en Streamlit) -- nunca en una variable global -- para que no se mezclen los datos
    de un usuario con los de otro que esté usando la app al mismo tiempo."""
    if df_subido is None or df_subido.empty:
        return DF_REFERENCIA
    combinado = pd.concat([DF_REFERENCIA, df_subido], ignore_index=True)
    combinado = combinado.sort_values(['CODIGO_ESTABLECIMIENTO', 'COD_AREA_FUNCIONAL', 'PERIODO', 'MES'])
    # si el archivo subido trae un mes que ya conocíamos de ese hospital, gana el archivo
    # subido (más probable que sea el dato más reciente) -- decisión "se completa con lo
    # ya conocido".
    combinado = combinado.drop_duplicates(
        subset=['CODIGO_ESTABLECIMIENTO', 'COD_AREA_FUNCIONAL', 'PERIODO', 'MES'], keep='last'
    )
    canon_hospitales, canon_areas = _canon_por_codigo(df_subido)
    return _aplicar_canon(combinado, canon_hospitales, canon_areas)


def hospitales_disponibles(df_subido):
    """Universo de nombres de hospital para el dropdown/selectbox: solo los del archivo
    subido si hay uno, o los 208 de referencia si no se ha subido nada. Un hospital
    aparece una sola vez aunque el archivo traiga varias grafías de su nombre."""
    if df_subido is not None and not df_subido.empty:
        canon_hospitales, _ = _canon_por_codigo(df_subido)
        nombres = df_subido['CODIGO_ESTABLECIMIENTO'].map(canon_hospitales).fillna(df_subido['ESTABLECIMIENTO'])
        return sorted(nombres.unique().tolist())
    return HOSPITALES_REFERENCIA


def codigo_de_hospital(df_activo, nombre_hospital):
    """CODIGO_ESTABLECIMIENTO del hospital elegido en el selector, o None si ese nombre
    no está en los datos activos. Es el puente entre el nombre que ve el usuario y el
    código que usan areas_disponibles() y el resto de la lógica."""
    filas = df_activo[df_activo['ESTABLECIMIENTO'] == nombre_hospital]
    return int(filas['CODIGO_ESTABLECIMIENTO'].iloc[0]) if not filas.empty else None


def areas_disponibles(df_activo, establecimiento_cod):
    """Áreas funcionales que ese hospital realmente tiene en los datos activos, ordenadas
    alfabéticamente. Devuelve [] si el código no aparece.

    Cómo conectarla en la interfaz (cascade hospital -> área):

        df_activo = logica.dataframe_activo(df_subido)
        cod = logica.codigo_de_hospital(df_activo, hospital_elegido)
        areas = logica.areas_disponibles(df_activo, cod) if cod else []

    En Streamlit basta con pasar `areas` al st.selectbox de área en vez de
    logica.AREAS_REFERENCIA; como el script se re-ejecuta al cambiar el hospital, la lista
    se actualiza sola. En Gradio hay que colgar un hospital_dd.change(...) que devuelva
    gr.update(choices=areas, value=areas[0] if areas else None).

    Si el hospital no tiene ninguna área (lista vacía), conviene deshabilitar el botón de
    predecir: predecir_valor() respondería 'No hay historial real ...' de todos modos."""
    if establecimiento_cod is None:
        return []
    filas = df_activo[df_activo['CODIGO_ESTABLECIMIENTO'] == establecimiento_cod]
    return sorted(filas['AREA_FUNCIONAL'].dropna().unique().tolist())


def buscar_hospital(texto, df_subido):
    """Usa sugerir_coincidencias (acepta nombres coloquiales, sin tildes, incompletos)
    en vez del filtro literal de substring de los widgets nativos."""
    universo = hospitales_disponibles(df_subido)
    if not texto:
        return universo
    return sugerir_coincidencias(texto, universo, n=8)


def cargar_archivo_subido(ruta_o_buffer):
    """Envuelve cargar_csv_hospital devolviendo (df_subido, mensaje, es_error) en vez
    de lanzar la excepción directamente, para que ambas interfaces puedan mostrar el
    mensaje sin repetir el try/except."""
    try:
        df_subido = cargar_csv_hospital(ruta_o_buffer)
    except (ColumnasFaltantesError, ArchivoVacioError) as e:
        return None, str(e), True

    hospitales_subidos = sorted(df_subido['ESTABLECIMIENTO'].unique().tolist())
    filas_descartadas = df_subido.attrs.get('filas_invalidas_descartadas', 0)
    aviso = f' ({filas_descartadas} fila(s) descartadas por datos inválidos.)' if filas_descartadas else ''
    mensaje = f'Archivo cargado: {len(df_subido)} filas · {len(hospitales_subidos)} hospital(es).{aviso}'
    return df_subido, mensaje, False


def _sumar_meses(anio, mes, n):
    idx = anio * 12 + (mes - 1) + n
    return idx // 12, idx % 12 + 1


def _fila_hospital(df_activo, nombre_hospital):
    fila = df_activo[df_activo['ESTABLECIMIENTO'] == nombre_hospital]
    return fila.iloc[0] if not fila.empty else None


def _codigo_area(nombre_area, df_activo=None):
    """Se busca en los datos activos y no solo en la referencia: así un área que venga
    del archivo subido tampoco queda 'no reconocida'."""
    df = DF_REFERENCIA if df_activo is None else df_activo
    fila = df[df['AREA_FUNCIONAL'] == nombre_area]
    return int(fila['COD_AREA_FUNCIONAL'].iloc[0]) if not fila.empty else None


def _camas_y_complejidad(df_activo, establecimiento_cod, area_cod):
    """Promedio de camas del área elegida y complejidad del hospital, calculados a
    partir de los propios datos activos (referencia + lo subido) -- así funciona
    también para un hospital que el modelo nunca vio entrenando, en vez de depender
    de un valor por defecto cuando no está en los diccionarios aprendidos."""
    filas_hosp = df_activo[df_activo['CODIGO_ESTABLECIMIENTO'] == establecimiento_cod]
    filas_area = filas_hosp[filas_hosp['COD_AREA_FUNCIONAL'] == area_cod]
    camas_area = filas_area['PROMEDIO_CAMAS_DISPONIBLE'].mean()
    if pd.isna(camas_area):
        camas_area = None

    if establecimiento_cod in ARTEFACTOS['mapa_complejidad']:
        complejidad = None  # deja que predecir_recursivo use el valor aprendido en entrenamiento
    else:
        camas_hosp_mes = filas_hosp.groupby(['PERIODO', 'MES'])['PROMEDIO_CAMAS_DISPONIBLE'].sum()
        tamano_hosp = camas_hosp_mes.mean() if not camas_hosp_mes.empty else np.nan
        cortes = ARTEFACTOS['cortes_complejidad']
        complejidad = clasificar_complejidad(tamano_hosp, cortes)

    return camas_area, complejidad


def predecir_valor(hospital_nombre, area_nombre, horizonte_label, df_subido):
    """Función central que usan las dos interfaces. Devuelve un diccionario plano
    (nunca HTML ni objetos de un framework de UI en particular):
    {'ok': bool, 'mensaje': str, 'es_error': bool, 'valor', 'semaforo', 'trayectoria',
     'es_prediccion', 'anio_obj', 'mes_obj'} -- las claves de predicción solo están
    presentes cuando ok=True."""
    if not hospital_nombre or not area_nombre:
        return {'ok': False, 'es_error': False, 'mensaje': 'Elige un hospital y un área funcional.'}

    df_activo = dataframe_activo(df_subido)
    fila_hosp = _fila_hospital(df_activo, hospital_nombre)
    if fila_hosp is None:
        return {'ok': False, 'es_error': True, 'mensaje': f'No hay datos cargados para "{hospital_nombre}".'}

    area_cod = _codigo_area(area_nombre, df_activo)
    if area_cod is None:
        return {'ok': False, 'es_error': True, 'mensaje': f'Área funcional "{area_nombre}" no reconocida.'}

    establecimiento_cod = int(fila_hosp['CODIGO_ESTABLECIMIENTO'])
    glosa_sss = fila_hosp['GLOSA_SSS']

    serie = construir_serie_historica(df_activo, establecimiento_cod, area_cod)
    if not serie:
        return {
            'ok': False, 'es_error': True,
            'mensaje': (
                f'No hay historial real de "{area_nombre}" para {hospital_nombre}. '
                'Sube un archivo con al menos 1 mes de datos de esa área antes de predecir.'
            ),
        }

    ultimo_anio, ultimo_mes, _ = serie[-1]
    horizonte = HORIZONTES.get(horizonte_label)
    if horizonte is None:
        # sin valor por defecto: una etiqueta desconocida (ej. el antiguo '6 meses') se
        # informa en vez de predecir en silencio a otro horizonte
        return {
            'ok': False, 'es_error': True,
            'mensaje': f'Horizonte "{horizonte_label}" no válido. Opciones: {", ".join(HORIZONTES)}.',
        }
    anio_obj, mes_obj = _sumar_meses(ultimo_anio, ultimo_mes, horizonte)
    camas_area, complejidad = _camas_y_complejidad(df_activo, establecimiento_cod, area_cod)

    try:
        valor, trayectoria, es_prediccion = predecir_recursivo(
            MODELO, FEATURES, ARTEFACTOS, df_activo,
            establecimiento_cod, area_cod, glosa_sss, area_nombre,
            anio_obj, mes_obj,
            promedio_camas_disponible=camas_area, complejidad_override=complejidad,
        )
    except (SinHistorialError, MesFueraDeRangoError, HorizonteFueraDeRangoError) as e:
        return {'ok': False, 'es_error': True, 'mensaje': str(e)}

    return {
        'ok': True, 'es_error': False, 'mensaje': '',
        'valor': valor, 'semaforo': evaluar_semaforo(valor), 'trayectoria': trayectoria,
        'es_prediccion': es_prediccion, 'anio_obj': anio_obj, 'mes_obj': mes_obj,
    }


def graficar_trayectoria(trayectoria):
    """Figura de matplotlib con la trayectoria real + proyectada -- se muestra igual
    en Gradio (gr.Plot) y en Streamlit (st.pyplot)."""
    etiquetas = [f'{mes:02d}/{str(anio)[2:]}' for (anio, mes, _, _) in trayectoria]
    valores = [v for (_, _, v, _) in trayectoria]
    n_reales = sum(1 for (_, _, _, es_real) in trayectoria if es_real)
    x = list(range(len(trayectoria)))

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(x[:n_reales], valores[:n_reales], marker='o', color='#0e6e64', linewidth=2.2, label='Real')
    if n_reales < len(trayectoria):
        borde = max(n_reales - 1, 0)
        ax.plot(x[borde:], valores[borde:], marker='o', linestyle='--', color='#0e6e64',
                linewidth=2.2, markerfacecolor='white', label='Proyectado')
    for banda, color in [(60, '#0ca30c'), (80, '#9aa0a6'), (90, '#d03b3b')]:
        ax.axhline(banda, color=color, linestyle=':', linewidth=1, alpha=.7)
        ax.text(len(trayectoria) - 0.4, banda + 1.5, f'{banda}%', fontsize=7, color=color, ha='right')
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=8)
    ax.set_ylabel('Índice ocupacional (%)')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    for lado in ('top', 'right'):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    return fig
