"""
Pipeline de features V3 (Avance 05) — sin identidad de hospital, sin fuga temporal.
Extiende el diseno de Avance 04, sacando las variables de fuga que Sebastian identifico
y que Avance 04 todavia no habia sacado: DIAS_ESTADA, NUMERO_EGRESOS, EGRESOS_FALLECIDOS,
TRASLADOS, PROMEDIO_DIAS_ESTADA, LETALIDAD (describen el mismo mes que se predice).
"""
import pandas as pd
import numpy as np

GROUP_KEYS = ['CODIGO_ESTABLECIMIENTO', 'COD_AREA_FUNCIONAL']

LEAKAGE_MATEMATICA = ['DIAS_CAMAS_OCUPADAS', 'DIAS_CAMAS_DISPONIBLES', 'INDICE_OCUPACIONAL', 'INDICE_ROTACION']
LEAKAGE_TEMPORAL_NOWCASTING = ['DIAS_ESTADA', 'NUMERO_EGRESOS', 'EGRESOS_FALLECIDOS',
                                'TRASLADOS', 'PROMEDIO_DIAS_ESTADA', 'LETALIDAD']
IDENTITY_COLS = ['CODIGO_ESTABLECIMIENTO', 'TIPO_PERTENENCIA', 'COD_SSS', 'ESTABLECIMIENTO']

FEATURES_V3 = [
    'PERIODO', 'MES_SIN', 'MES_COS',
    'PROMEDIO_CAMAS_DISPONIBLE',
    'NIVEL_COMPLEJIDAD_PROXY', 'GLOSA_SSS_ENC', 'AREA_FUNCIONAL_ENC',
    'LAG1_OCUPACION', 'LAG2_OCUPACION', 'ROLL3_OCUPACION',
]


def compute_lag_features(df, group_keys, target_col, fill_value=None):
    """LAG1, LAG2 y ROLL3 de target_col, agrupado por group_keys.
    Reutilizable en entrenamiento y en inferencia (prediccion recursiva)."""
    df = df.copy()
    grouped = df.groupby(group_keys)[target_col]
    df['LAG1_OCUPACION'] = grouped.shift(1)
    df['LAG2_OCUPACION'] = grouped.shift(2)
    df['ROLL3_OCUPACION'] = grouped.transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    if fill_value is not None:
        for col in ['LAG1_OCUPACION', 'LAG2_OCUPACION', 'ROLL3_OCUPACION']:
            df[col] = df[col].fillna(fill_value)
    return df


def clasificar_complejidad(tamano, cortes):
    if pd.isna(tamano):
        return 1
    if tamano <= cortes[0]:
        return 0
    elif tamano <= cortes[1]:
        return 1
    return 2


def construir_complejidad_proxy(df, train_mask):
    """Tercil de tamano en camas por hospital, cortes aprendidos SOLO en train."""
    camas_hosp_mes = df.groupby(['CODIGO_ESTABLECIMIENTO', 'PERIODO', 'MES'])['PROMEDIO_CAMAS_DISPONIBLE'].sum()

    periodos_train = df.loc[train_mask, ['CODIGO_ESTABLECIMIENTO', 'PERIODO', 'MES']].drop_duplicates()
    camas_train = camas_hosp_mes.reset_index().merge(periodos_train, on=['CODIGO_ESTABLECIMIENTO', 'PERIODO', 'MES'], how='inner')
    tamano_promedio_train = camas_train.groupby('CODIGO_ESTABLECIMIENTO')['PROMEDIO_CAMAS_DISPONIBLE'].mean()
    cortes = tamano_promedio_train.quantile([1 / 3, 2 / 3]).values

    # tamano propio de cada hospital (incluidos los que solo aparecen fuera de train), como atributo estructural
    tamano_promedio_todos = camas_hosp_mes.groupby('CODIGO_ESTABLECIMIENTO').mean()
    mapa = tamano_promedio_todos.apply(lambda t: clasificar_complejidad(t, cortes)).to_dict()
    return mapa, cortes


def m_estimate_encoding(df_train, dfs_otros, col, target_col, m=10):
    global_mean = df_train[target_col].mean()
    stats = df_train.groupby(col)[target_col].agg(['sum', 'count'])
    stats['encoded'] = (stats['sum'] + m * global_mean) / (stats['count'] + m)
    mapping = stats['encoded'].to_dict()
    encoded_col = col + '_ENC'
    df_train[encoded_col] = df_train[col].map(mapping).fillna(global_mean)
    for d in dfs_otros:
        d[encoded_col] = d[col].map(mapping).fillna(global_mean)
    return mapping, global_mean


def construir_features(df, train_mask):
    """Aplica todo el pipeline (complejidad, encodings, lags, seasonality) sobre df ya
    ordenado cronologicamente, devolviendo el df enriquecido + los artefactos aprendidos
    en train (para poder reaplicarlos despues en inferencia sin fuga)."""
    df = df.copy()

    mapa_complejidad, cortes = construir_complejidad_proxy(df, train_mask)
    df['NIVEL_COMPLEJIDAD_PROXY'] = df['CODIGO_ESTABLECIMIENTO'].map(mapa_complejidad).fillna(1).astype(int)

    df['MES_SIN'] = np.sin(2 * np.pi * df['MES'] / 12)
    df['MES_COS'] = np.cos(2 * np.pi * df['MES'] / 12)

    media_global_train = df.loc[train_mask, 'TARGET'].mean()
    df = compute_lag_features(df, GROUP_KEYS, 'TARGET', fill_value=media_global_train)

    df_train = df[train_mask].copy()
    df_resto = df[~train_mask].copy()
    mapping_zona, _ = m_estimate_encoding(df_train, [df_resto], 'GLOSA_SSS', 'TARGET', m=10)
    mapping_area, _ = m_estimate_encoding(df_train, [df_resto], 'AREA_FUNCIONAL', 'TARGET', m=10)
    df = pd.concat([df_train, df_resto]).sort_index()

    artefactos = {
        'mapa_complejidad': mapa_complejidad,
        'cortes_complejidad': cortes.tolist(),
        'media_global_train': media_global_train,
        'mapping_zona': mapping_zona,
        'mapping_area': mapping_area,
        'global_mean_zona': df_train['TARGET'].mean(),
    }
    return df, artefactos
