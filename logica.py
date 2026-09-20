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

from carga_datos import (
    ArchivoIlegibleError,
    ArchivoVacioError,
    ColumnasFaltantesError,
    cargar_csv_hospital,
)
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


# ── Métricas de validación del modelo, leídas del .pkl. Son números, no textos: la
#    interfaz decide cómo mostrarlos. None si el .pkl no las trae. ─────────────────
_metricas_test = _artefacto.get('metricas_test') or {}
_por_horizonte = (_artefacto.get('metricas_por_horizonte') or {}).get('horizontes') or {}

MAE = _metricas_test.get('MAE')  # 7.358 pp
R2 = _metricas_test.get('R2')    # 0.862

# Lo mismo por horizonte, con la etiqueta de HORIZONTES como clave ('1 mes', '2 meses',
# '3 meses'). Son el error de la predicción recursiva, que es la que hace la app, y es
# mayor que el de un paso: MAE 7.84 / 9.25 / 10.08 pp.
MAE_POR_HORIZONTE = {etiqueta: _por_horizonte[str(meses)]['MAE']
                     for etiqueta, meses in HORIZONTES.items() if str(meses) in _por_horizonte}
R2_POR_HORIZONTE = {etiqueta: _por_horizonte[str(meses)]['R2']
                    for etiqueta, meses in HORIZONTES.items() if str(meses) in _por_horizonte}


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


def ultimo_periodo(df_activo):
    """(anio, mes) del dato más reciente de todos los datos activos. Es la fecha de corte
    contra la que se compara cada área para saber si está al día."""
    idx = int((df_activo['PERIODO'] * 12 + df_activo['MES'] - 1).max())
    return idx // 12, idx % 12 + 1


def anio_vigente(df_activo):
    """Último año con datos. Un hospital o un área sin datos de este año se considera
    descontinuado y no se ofrece en los selectores: predecir desde su último mes real
    devolvería una 'proyección' de un mes que ya pasó (ej. hospitales de campaña que
    dejaron de reportar en 2020). No está fijo en 2026: si el archivo subido trae un año
    más nuevo, ese pasa a ser el vigente."""
    return int(df_activo['PERIODO'].max())


def _con_datos_vigentes(df_activo):
    """Filas del año vigente, que son las que definen qué se ofrece en los selectores."""
    return df_activo[df_activo['PERIODO'] == anio_vigente(df_activo)]


def hospitales_disponibles(df_subido, df_activo=None):
    """Universo de nombres de hospital para el dropdown/selectbox: solo los del archivo
    subido si hay uno, o los de referencia si no se ha subido nada. Un hospital aparece
    una sola vez aunque el archivo traiga varias grafías de su nombre, y solo si tiene
    datos del año vigente (ver anio_vigente).

    df_activo es opcional: si la interfaz ya lo tiene armado conviene pasarlo para no
    recalcularlo en cada interacción."""
    if df_activo is None:
        df_activo = dataframe_activo(df_subido)
    vigentes = _con_datos_vigentes(df_activo)
    if df_subido is not None and not df_subido.empty:
        codigos = set(df_subido['CODIGO_ESTABLECIMIENTO'])
        vigentes = vigentes[vigentes['CODIGO_ESTABLECIMIENTO'].isin(codigos)]
    return sorted(vigentes['ESTABLECIMIENTO'].dropna().unique().tolist())


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

    Solo se devuelven las áreas con datos del año vigente: un área que dejó de reportar
    antes (ej. hasta 2014) no se ofrece, aunque el hospital siga activo en otras áreas.
    Para saber hasta qué mes llega cada una, usar cobertura_area().

    Si el hospital no tiene ninguna área (lista vacía), conviene deshabilitar el botón de
    predecir: predecir_valor() respondería 'No hay historial real ...' de todos modos."""
    if establecimiento_cod is None:
        return []
    filas = _con_datos_vigentes(df_activo)
    filas = filas[filas['CODIGO_ESTABLECIMIENTO'] == establecimiento_cod]
    return sorted(filas['AREA_FUNCIONAL'].dropna().unique().tolist())


def cobertura_area(df_activo, establecimiento_cod, area_cod):
    """Desde y hasta qué mes hay datos reales de ese hospital+área, para que la interfaz
    avise cuando un área no llega al último mes del dataset. Devuelve None si no hay datos.

        {'desde': '01/2014', 'hasta': '05/2026', 'corte_datos': '06/2026',
         'al_dia': False, 'meses_de_atraso': 1,
         'desde_anio': 2014, 'desde_mes': 1, 'hasta_anio': 2026, 'hasta_mes': 5}

    'al_dia' es False cuando el área se queda antes del último mes de los datos activos
    (incluido lo que suba el usuario). Los textos los arma la interfaz."""
    filas = df_activo[(df_activo['CODIGO_ESTABLECIMIENTO'] == establecimiento_cod) &
                      (df_activo['COD_AREA_FUNCIONAL'] == area_cod)]
    if filas.empty:
        return None
    idx = filas['PERIODO'] * 12 + filas['MES'] - 1
    primero, ultimo = int(idx.min()), int(idx.max())
    corte_anio, corte_mes = ultimo_periodo(df_activo)
    corte = corte_anio * 12 + corte_mes - 1
    fmt = lambda i: f'{i % 12 + 1:02d}/{i // 12}'
    return {
        'desde': fmt(primero), 'hasta': fmt(ultimo), 'corte_datos': fmt(corte),
        'al_dia': ultimo >= corte, 'meses_de_atraso': corte - ultimo,
        'desde_anio': primero // 12, 'desde_mes': primero % 12 + 1,
        'hasta_anio': ultimo // 12, 'hasta_mes': ultimo % 12 + 1,
    }


def buscar_hospital(texto, df_subido, df_activo=None):
    """Usa sugerir_coincidencias (acepta nombres coloquiales, sin tildes, incompletos)
    en vez del filtro literal de substring de los widgets nativos."""
    universo = hospitales_disponibles(df_subido, df_activo)
    if not texto:
        return universo
    return sugerir_coincidencias(texto, universo, n=8)


def cargar_archivo_subido_detallado(ruta_o_buffer):
    """Envuelve cargar_csv_hospital sin lanzar excepciones, para que las interfaces no
    repitan el try/except. Devuelve:

        {'df': DataFrame o None,
         'mensaje': texto para mostrarle al usuario,
         'es_error': bool,
         'codigo': None | 'estructura' | 'ilegible',
         'detalle_tecnico': '' o el detalle exacto del fallo}

    'mensaje' nunca trae jerga ni nombres de columnas: al usuario no le sirve saber qué
    columna falta, porque el formato lo define el REM. El detalle exacto viaja aparte en
    'detalle_tecnico', por si la interfaz quiere ofrecerlo en un expander opcional o
    dejarlo en un log."""
    try:
        df_subido = cargar_csv_hospital(ruta_o_buffer)
    except (ColumnasFaltantesError, ArchivoVacioError, ArchivoIlegibleError) as e:
        return {'df': None, 'mensaje': e.mensaje_usuario, 'es_error': True,
                'codigo': e.codigo, 'detalle_tecnico': str(e)}

    hospitales_subidos = sorted(df_subido['ESTABLECIMIENTO'].unique().tolist())
    filas_descartadas = df_subido.attrs.get('filas_invalidas_descartadas', 0)
    aviso = f' ({filas_descartadas} fila(s) descartadas por datos inválidos.)' if filas_descartadas else ''
    mensaje = f'Archivo cargado: {len(df_subido)} filas · {len(hospitales_subidos)} hospital(es).{aviso}'
    return {'df': df_subido, 'mensaje': mensaje, 'es_error': False,
            'codigo': None, 'detalle_tecnico': ''}


def cargar_archivo_subido(ruta_o_buffer):
    """(df_subido, mensaje, es_error) -- la forma corta, que es la que usan hoy las dos
    interfaces. Para el detalle técnico del fallo, usar cargar_archivo_subido_detallado()."""
    r = cargar_archivo_subido_detallado(ruta_o_buffer)
    return r['df'], r['mensaje'], r['es_error']


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
        'establecimiento_cod': establecimiento_cod, 'area_cod': area_cod,
    }


# ── Motor de recomendación de traslado ────────────────────────────────────────────

UMBRAL_TRASLADO = 85.0  # umbral empírico de ICOVID Chile, el mismo del semáforo amarillo


def _servicio_por_hospital(df_activo):
    """{CODIGO_ESTABLECIMIENTO: servicio de salud}, con el nombre más reciente de la GLOSA.

    El .parquet no trae COD_SSS, y tres servicios cambiaron de nombre (Arica -> Arica y
    Parinacota, Iquique -> Tarapacá, Valdivia -> Los Ríos). Tomando el nombre más reciente
    de cada hospital, agrupar por este diccionario da exactamente los mismos 29 grupos que
    agrupar por COD_SSS para los hospitales vigentes (verificado sobre el CSV del REM).

    Ojo: esto es solo para agrupar candidatos. La predicción sigue usando la GLOSA de la
    primera fila del hospital, que es el nombre que conoce el modelo."""
    d = df_activo.assign(_ORDEN=df_activo['PERIODO'] * 12 + df_activo['MES'])
    d = d.sort_values('_ORDEN', kind='stable')
    return d.groupby('CODIGO_ESTABLECIMIENTO')['GLOSA_SSS'].last().to_dict()


def _predecir_par(df_activo, establecimiento_cod, area_cod, anio_obj, mes_obj):
    """Índice proyectado de un hospital+área para un mes objetivo concreto, con el mismo
    camino que predecir_valor(). Devuelve None si esa área no puede llegar a ese mes (por
    ejemplo, su último dato está a más de HORIZONTE_MAXIMO meses)."""
    filas_hosp = df_activo[df_activo['CODIGO_ESTABLECIMIENTO'] == establecimiento_cod]
    if filas_hosp.empty:
        return None
    area_nombre = filas_hosp.loc[filas_hosp['COD_AREA_FUNCIONAL'] == area_cod, 'AREA_FUNCIONAL']
    if area_nombre.empty:
        return None
    camas_area, complejidad = _camas_y_complejidad(df_activo, establecimiento_cod, area_cod)
    try:
        valor, _, _ = predecir_recursivo(
            MODELO, FEATURES, ARTEFACTOS, df_activo,
            establecimiento_cod, area_cod, filas_hosp['GLOSA_SSS'].iloc[0], area_nombre.iloc[0],
            anio_obj, mes_obj, promedio_camas_disponible=camas_area, complejidad_override=complejidad,
        )
    except (SinHistorialError, MesFueraDeRangoError, HorizonteFueraDeRangoError):
        return None
    return valor


def sugerir_traslado(df_activo, establecimiento_cod, area_cod, anio_obj, mes_obj,
                     umbral=UMBRAL_TRASLADO, maximo=3):
    """Alternativas de derivación cuando un área queda sobre el umbral (85%).

    Busca en el mismo servicio de salud (equivalente a COD_SSS) los otros hospitales que
    para ese mismo mes objetivo proyectan menos de `umbral`, y devuelve hasta `maximo`.
    Primero los que tienen la misma área consultada, ordenados de menor a mayor ocupación;
    si no alcanzan, se completan con hospitales que tengan otras áreas bajo el umbral.
    Dentro de cada hospital se listan todas sus áreas bajo el umbral, con la misma área
    consultada primero.

    Devuelve:
        {'servicio': 'Metropolitano Oriente', 'umbral': 85.0, 'anio_obj': 2026, 'mes_obj': 9,
         'hospitales': [{'codigo': 112100, 'nombre': '...', 'misma_area': True,
                         'areas': [{'area': '...', 'valor': 62.3, 'misma_area': True}, ...]}],
         'pares_evaluados': 38, 'sin_alternativas': False}

    Con 'sin_alternativas' en True no hay dónde derivar dentro del servicio: ahí
    corresponde evaluar altas a domicilio según criticidad. El texto lo pone la interfaz."""
    servicios = _servicio_por_hospital(df_activo)
    servicio = servicios.get(establecimiento_cod)
    vigentes = _con_datos_vigentes(df_activo)
    candidatos = vigentes[vigentes['CODIGO_ESTABLECIMIENTO'].map(servicios) == servicio]
    candidatos = candidatos[candidatos['CODIGO_ESTABLECIMIENTO'] != establecimiento_cod]

    por_hospital, evaluados = {}, 0
    for (cod, cod_area), filas in candidatos.groupby(['CODIGO_ESTABLECIMIENTO', 'COD_AREA_FUNCIONAL']):
        evaluados += 1
        valor = _predecir_par(df_activo, cod, cod_area, anio_obj, mes_obj)
        if valor is None or valor >= umbral:
            continue
        datos = por_hospital.setdefault(cod, {'codigo': int(cod), 'nombre': filas['ESTABLECIMIENTO'].iloc[0],
                                              'misma_area': False, 'areas': []})
        es_misma = cod_area == area_cod
        datos['areas'].append({'area': filas['AREA_FUNCIONAL'].iloc[0], 'valor': float(valor),
                               'misma_area': es_misma})
        datos['misma_area'] = datos['misma_area'] or es_misma

    for datos in por_hospital.values():
        datos['areas'].sort(key=lambda a: (not a['misma_area'], a['valor']))
        datos['mejor_valor'] = datos['areas'][0]['valor']
        datos['valor_misma_area'] = next((a['valor'] for a in datos['areas'] if a['misma_area']), None)

    # los que tienen la misma área van primero, ordenados por esa área; el resto, por su
    # área más desocupada
    orden = sorted(por_hospital.values(),
                   key=lambda h: (not h['misma_area'],
                                  h['valor_misma_area'] if h['misma_area'] else h['mejor_valor']))
    elegidos = orden[:maximo]
    return {
        'servicio': servicio, 'umbral': umbral, 'anio_obj': anio_obj, 'mes_obj': mes_obj,
        'hospitales': elegidos, 'pares_evaluados': evaluados, 'sin_alternativas': not elegidos,
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
