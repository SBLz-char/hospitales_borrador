"""
Carga y validacion del CSV que sube el usuario en la interfaz. Se detiene con un
mensaje claro (listando exactamente que falta) en vez de intentar adivinar o rellenar
columnas ausentes -- la version de prueba solo soporta CSV (separador ';', igual al
formato oficial de REM 20); Excel queda para una version futura.
"""
import numpy as np
import pandas as pd

# columnas que tienen que venir siempre, sin las cuales no se puede identificar el
# hospital/area ni construir las features del modelo
COLUMNAS_OBLIGATORIAS = [
    'CODIGO_ESTABLECIMIENTO', 'ESTABLECIMIENTO',
    'COD_AREA_FUNCIONAL', 'AREA_FUNCIONAL',
    'GLOSA_SSS', 'PERIODO', 'MES', 'PROMEDIO_CAMAS_DISPONIBLE',
]

# para el target hace falta INDICE_OCUPACIONAL ya calculado, o los dos componentes
# para calcularlo (dias-cama ocupados / dias-cama disponibles)
COLUMNAS_TARGET_DIRECTO = ['INDICE_OCUPACIONAL']
COLUMNAS_TARGET_DERIVADO = ['DIAS_CAMAS_OCUPADAS', 'DIAS_CAMAS_DISPONIBLES']


class ColumnasFaltantesError(Exception):
    """El archivo no trae las columnas minimas necesarias. El mensaje ya viene armado
    con la lista exacta de lo que falta, lista para mostrar en la interfaz."""
    pass


class ArchivoVacioError(Exception):
    """El archivo se pudo leer pero no tiene filas de datos."""
    pass


def _leer_csv(ruta_o_buffer):
    """Lee el CSV con el separador oficial de REM 20 (';'). Los CSV que exportan los
    hospitales suelen venir en Latin-1 (tildes/enes rompen en UTF-8 estricto); se
    intenta UTF-8 primero por ser el estandar mas comun hoy, y se cae a Latin-1 si falla."""
    try:
        return pd.read_csv(ruta_o_buffer, sep=';', encoding='utf-8')
    except UnicodeDecodeError:
        if hasattr(ruta_o_buffer, 'seek'):
            ruta_o_buffer.seek(0)
        return pd.read_csv(ruta_o_buffer, sep=';', encoding='latin-1')


def validar_columnas(df):
    """Devuelve la lista de columnas obligatorias que faltan, mas una nota si tampoco
    hay forma de construir el TARGET. Lista vacia = todo bien."""
    faltantes = [c for c in COLUMNAS_OBLIGATORIAS if c not in df.columns]

    tiene_target_directo = all(c in df.columns for c in COLUMNAS_TARGET_DIRECTO)
    tiene_target_derivado = all(c in df.columns for c in COLUMNAS_TARGET_DERIVADO)
    if not tiene_target_directo and not tiene_target_derivado:
        faltantes.append('INDICE_OCUPACIONAL (o, en su defecto, DIAS_CAMAS_OCUPADAS y DIAS_CAMAS_DISPONIBLES)')

    return faltantes


def _construir_target(df):
    if 'INDICE_OCUPACIONAL' in df.columns:
        target = pd.to_numeric(df['INDICE_OCUPACIONAL'], errors='coerce')
    else:
        ocupadas = pd.to_numeric(df['DIAS_CAMAS_OCUPADAS'], errors='coerce')
        disponibles = pd.to_numeric(df['DIAS_CAMAS_DISPONIBLES'], errors='coerce')
        target = np.where(disponibles > 0, ocupadas / disponibles * 100, np.nan)
        target = pd.Series(target, index=df.index)
    # mismo techo fisico (capping a 100%) validado y aplicado en Avance 03/04/05
    return target.clip(upper=100)


def cargar_csv_hospital(ruta_o_buffer):
    """Lee, valida y normaliza el CSV subido por el usuario. Devuelve un DataFrame
    con las columnas minimas que el resto del pipeline espera (mismos nombres que
    dataset_referencia.parquet), listo para concatenar con el historial conocido.

    Lanza ColumnasFaltantesError con el detalle exacto de que falta, o
    ArchivoVacioError si el archivo no trae filas."""
    df = _leer_csv(ruta_o_buffer)

    if df.empty:
        raise ArchivoVacioError('El archivo no tiene filas de datos.')

    faltantes = validar_columnas(df)
    if faltantes:
        raise ColumnasFaltantesError(
            'Al archivo le faltan columnas necesarias: ' + ', '.join(faltantes) + '. '
            'Revisa que sea un extracto con el mismo formato del REM 20 (separador ";").'
        )

    df = df.copy()
    # el CSV oficial del REM trae espacios sobrantes en los nombres (en el extracto de
    # julio 2026, 156.918 de 166.318 filas tienen el area con un espacio al final). Sin
    # limpiarlos, 'Area X ' != 'Area X': el nombre no calza con el que aprendio el modelo
    # y AREA_FUNCIONAL_ENC cae al promedio general sin avisar, ademas de duplicar opciones
    # en los selectores de la interfaz.
    for col in ['ESTABLECIMIENTO', 'AREA_FUNCIONAL', 'GLOSA_SSS']:
        df[col] = df[col].astype(str).str.strip()

    # se mantienen como enteros (mismo dtype que dataset_referencia.parquet) para que
    # las claves de agrupacion/merge calcen exactas, no como texto
    df['CODIGO_ESTABLECIMIENTO'] = pd.to_numeric(df['CODIGO_ESTABLECIMIENTO'], errors='coerce').astype('Int64')
    df['COD_AREA_FUNCIONAL'] = pd.to_numeric(df['COD_AREA_FUNCIONAL'], errors='coerce').astype('Int64')
    df['PERIODO'] = pd.to_numeric(df['PERIODO'], errors='coerce').astype('Int64')
    df['MES'] = pd.to_numeric(df['MES'], errors='coerce').astype('Int64')
    df['PROMEDIO_CAMAS_DISPONIBLE'] = pd.to_numeric(df['PROMEDIO_CAMAS_DISPONIBLE'], errors='coerce')
    df['TARGET'] = _construir_target(df)

    filas_antes = len(df)
    claves = ['CODIGO_ESTABLECIMIENTO', 'COD_AREA_FUNCIONAL', 'PERIODO', 'MES', 'TARGET']
    df = df.dropna(subset=claves)
    filas_invalidas = filas_antes - len(df)

    if df.empty:
        raise ArchivoVacioError(
            'Ninguna fila quedó con identificadores, período y ocupación válidos después de leer el archivo.'
        )

    for col in ['CODIGO_ESTABLECIMIENTO', 'COD_AREA_FUNCIONAL', 'PERIODO', 'MES']:
        df[col] = df[col].astype('int64')

    columnas_finales = [
        'CODIGO_ESTABLECIMIENTO', 'ESTABLECIMIENTO', 'COD_AREA_FUNCIONAL', 'AREA_FUNCIONAL',
        'GLOSA_SSS', 'PERIODO', 'MES', 'PROMEDIO_CAMAS_DISPONIBLE', 'TARGET',
    ]
    df = df[columnas_finales].sort_values(['CODIGO_ESTABLECIMIENTO', 'COD_AREA_FUNCIONAL', 'PERIODO', 'MES'])
    df.attrs['filas_invalidas_descartadas'] = int(filas_invalidas)
    return df
