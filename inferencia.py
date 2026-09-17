"""
Modulo de inferencia: prediccion recursiva de 1 a 3 meses + interfaz de ingreso de datos
+ capa prescriptiva de dos niveles.

Diseño: para predecir un mes objetivo mas alla del ultimo dato real conocido de un
hospital+area, se predice mes a mes desde el primer mes desconocido, reinyectando cada
prediccion como si fuera el nuevo LAG1 (estrategia recursiva) hasta llegar al horizonte
pedido. No se le pide al usuario TRASLADOS ni ningun otro dato de fuga temporal: solo
hospital, area y el mes/horizonte a proyectar.

El horizonte se limita a HORIZONTE_MAXIMO meses desde el ultimo dato real: desde el
mes 4 los tres rezagos (LAG1, LAG2, ROLL3) ya serian solo predicciones del propio
modelo, sin ningun dato real.
"""
import difflib
import math
import unicodedata
import numpy as np
import pandas as pd

HORIZONTE_MAXIMO = 3


class SinHistorialError(Exception):
    """No hay ningún mes real cargado para el hospital+área pedido -- no se puede
    calcular LAG1/LAG2/ROLL3 a partir de historial propio. Se prefiere bloquear con
    un mensaje claro antes que predecir en silencio con el promedio general de la
    red disfrazado de predicción personalizada."""
    pass


class MesFueraDeRangoError(Exception):
    """El mes pedido es anterior al primer dato real disponible para este hospital+área
    (no hay cómo saber ese valor, ni real ni proyectado)."""
    pass


class HorizonteFueraDeRangoError(Exception):
    """El mes pedido esta a mas de HORIZONTE_MAXIMO meses del ultimo dato real del
    hospital+área: habria que encadenar mas predicciones de las permitidas."""
    pass


def _indice_periodo(anio, mes):
    """Convierte (anio, mes) a un entero comparable cronológicamente."""
    return anio * 12 + mes


def _normalizar(txt):
    """Minusculas y sin tildes, para que el fuzzy matching no falle por acentos
    (ej. 'sotero del rio' -> 'Hospital Sótero del Río') -- limitacion real que
    tenia la interfaz original de la Plataforma_Predictiva de prueba."""
    txt = unicodedata.normalize('NFKD', txt.lower()).encode('ascii', 'ignore').decode('ascii')
    return txt.strip()


def periodo_siguiente(anio, mes):
    total = anio * 12 + (mes - 1) + 1
    return total // 12, (total % 12) + 1


def construir_serie_historica(df, establecimiento_cod, area_cod):
    """Serie ordenada (periodo, mes, target) real, para un hospital+area."""
    sub = df[(df['CODIGO_ESTABLECIMIENTO'] == establecimiento_cod) & (df['COD_AREA_FUNCIONAL'] == area_cod)]
    sub = sub.sort_values(['PERIODO', 'MES'])
    return list(zip(sub['PERIODO'], sub['MES'], sub['TARGET']))


def predecir_recursivo(modelo, features, artefactos, df_historico,
                        establecimiento_cod, area_cod, glosa_sss, area_funcional_nombre,
                        anio_obj, mes_obj, promedio_camas_disponible=None, complejidad_override=None):
    """Proyecta el indice ocupacional de (establecimiento, area) para (anio_obj, mes_obj),
    prediciendo recursivamente cada mes intermedio no observado. Devuelve (valor,
    trayectoria, es_prediccion): valor es el dato real si (anio_obj, mes_obj) ya estaba
    en el historial, o la proyeccion del modelo si es un mes futuro; trayectoria es la
    lista de (anio, mes, valor, es_real).

    Lanza SinHistorialError si el hospital+area no tiene ningun mes real cargado (no hay
    con que calcular LAG1 propio), MesFueraDeRangoError si el mes pedido es anterior al
    primer dato real disponible, y HorizonteFueraDeRangoError si el mes pedido esta a mas
    de HORIZONTE_MAXIMO meses del ultimo dato real."""
    serie = construir_serie_historica(df_historico, establecimiento_cod, area_cod)
    if not serie:
        raise SinHistorialError(
            'No hay historial real cargado para este hospital y area. '
            'Se necesita al menos 1 mes de datos reales para poder proyectar.'
        )

    ultimo_anio, ultimo_mes, _ = serie[-1]
    idx_objetivo = _indice_periodo(anio_obj, mes_obj)
    idx_ultimo = _indice_periodo(ultimo_anio, ultimo_mes)

    if idx_objetivo <= idx_ultimo:
        # el mes pedido ya es historia conocida (pasado o el ultimo mes real): se
        # devuelve el dato real en vez de "predecirlo" -- ademas, el bucle de abajo
        # nunca termina si se lo deja avanzar con un objetivo que no es futuro.
        for (a, m, v) in serie:
            if (a, m) == (anio_obj, mes_obj):
                trayectoria = [(a2, m2, v2, True) for (a2, m2, v2) in serie[-6:]]
                return v, trayectoria, False
        primer_anio, primer_mes, _ = serie[0]
        raise MesFueraDeRangoError(
            f'No hay dato real para {mes_obj}/{anio_obj}: es anterior al primer mes '
            f'cargado ({primer_mes}/{primer_anio}).'
        )

    if idx_objetivo - idx_ultimo > HORIZONTE_MAXIMO:
        raise HorizonteFueraDeRangoError(
            f'No se puede proyectar {mes_obj}/{anio_obj}: el ultimo dato real es '
            f'{ultimo_mes}/{ultimo_anio} y el horizonte maximo es {HORIZONTE_MAXIMO} meses.'
        )

    valores = [v for (_, _, v) in serie]
    trayectoria = [(a, m, v, True) for (a, m, v) in serie[-6:]]  # ultimos 6 reales, para mostrar contexto

    if promedio_camas_disponible is None:
        promedio_camas_disponible = artefactos.get('camas_por_hospital', {}).get(establecimiento_cod, np.nan)
    if promedio_camas_disponible is None or (isinstance(promedio_camas_disponible, float) and np.isnan(promedio_camas_disponible)):
        # hospital no visto en entrenamiento y sin dato de camas: usar NaN aca produce
        # predicciones sin sentido (ej. 0%) en vez de fallar de forma visible. Como
        # ultimo recurso se usa el corte bajo/mediano de complejidad como una cantidad
        # de camas "tipica" -- quien llama a esta funcion deberia pasar el promedio de
        # camas real del hospital (columna PROMEDIO_CAMAS_DISPONIBLE de sus propios
        # datos) en vez de depender de este fallback.
        cortes = artefactos.get('cortes_complejidad')
        promedio_camas_disponible = cortes[0] if cortes else 10.0

    if complejidad_override is not None:
        complejidad = complejidad_override
    else:
        complejidad = artefactos['mapa_complejidad'].get(establecimiento_cod, 1)
    zona_enc = artefactos['mapping_zona'].get(glosa_sss, artefactos['global_mean_zona'])
    area_enc = artefactos['mapping_area'].get(area_funcional_nombre, artefactos['global_mean_zona'])
    media_global = artefactos['media_global_train']

    cursor_anio, cursor_mes = ultimo_anio, ultimo_mes
    pred = None
    while (cursor_anio, cursor_mes) != (anio_obj, mes_obj):
        cursor_anio, cursor_mes = periodo_siguiente(cursor_anio, cursor_mes)

        lag1 = valores[-1] if len(valores) >= 1 else media_global
        lag2 = valores[-2] if len(valores) >= 2 else media_global
        roll3 = float(np.mean(valores[-3:])) if len(valores) >= 1 else media_global

        mes_sin = math.sin(2 * math.pi * cursor_mes / 12)
        mes_cos = math.cos(2 * math.pi * cursor_mes / 12)

        vector = pd.DataFrame([{
            'PERIODO': cursor_anio, 'MES_SIN': mes_sin, 'MES_COS': mes_cos,
            'PROMEDIO_CAMAS_DISPONIBLE': promedio_camas_disponible,
            'NIVEL_COMPLEJIDAD_PROXY': complejidad,
            'GLOSA_SSS_ENC': zona_enc, 'AREA_FUNCIONAL_ENC': area_enc,
            'LAG1_OCUPACION': lag1, 'LAG2_OCUPACION': lag2, 'ROLL3_OCUPACION': roll3,
        }])[features]

        pred = float(modelo.predict(vector)[0])
        pred = min(max(pred, 0.0), 100.0)  # mismo techo fisico usado en entrenamiento (capping a 100%)
        valores.append(pred)
        trayectoria.append((cursor_anio, cursor_mes, pred, False))

    return pred, trayectoria, True


def sugerir_coincidencias(valor, lista_valores, n=5, cutoff=0.4):
    """Nombres oficiales del REM son largos y formales (ej. 'Complejo Hospitalario Dr
    Sotero del Rio (Santiago, Puente Alto)'), asi que un difflib de string completo
    falla con busquedas coloquiales ('hospital sotero del rio'). Se prueba, en orden:
    exacto -> todas las palabras de la busqueda contenidas -> difflib de string completo."""
    valor_norm = _normalizar(valor)
    mapa_norm = {_normalizar(v): v for v in lista_valores}

    if valor_norm in mapa_norm:
        return [mapa_norm[valor_norm]]

    palabras = [p for p in valor_norm.split() if len(p) > 2 and p not in ('hospital', 'de', 'del', 'la', 'los')]
    if palabras:
        por_palabras = [k for k in mapa_norm if all(p in k for p in palabras)]
        if por_palabras:
            return [mapa_norm[k] for k in por_palabras[:n]]

    candidatos_norm = difflib.get_close_matches(valor_norm, list(mapa_norm.keys()), n=n, cutoff=cutoff)
    return [mapa_norm[k] for k in candidatos_norm]


def evaluar_semaforo(indice):
    """Capa prescriptiva de dos niveles: gubernamental/institucional (accion operativa) y
    ciudadano (lenguaje simple), sobre las mismas bandas ancladas al umbral empirico de
    ICOVID Chile (85%/90%)."""
    if indice <= 60:
        return {
            'color': '🟢 VERDE', 'nivel': 'Capacidad con holgura',
            'institucional': 'Programar mantenimientos preventivos y cirugías electivas complejas aprovechando la disponibilidad de camas.',
            'ciudadano': 'Este hospital tiene camas disponibles. La atención debería ser expedita.',
        }
    elif indice <= 80:
        return {
            'color': '🔵 AZUL', 'nivel': 'Operación regular',
            'institucional': 'Agilizar el giro de camas con altas tempranas (Discharge Before Noon) para mantener margen operativo.',
            'ciudadano': 'El hospital opera con normalidad. Pueden existir tiempos de espera habituales.',
        }
    elif indice <= 90:
        return {
            'color': '🟡 AMARILLO', 'nivel': 'Alerta de saturación inminente',
            'institucional': 'Trasladar pacientes estables a unidades de menor complejidad y liberar camas críticas.',
            'ciudadano': 'El hospital está cerca de su capacidad máxima. Es posible que la atención no urgente demore más de lo habitual.',
        }
    else:
        return {
            'color': '🔴 ROJO', 'nivel': 'Saturación crítica proyectada',
            'institucional': 'Habilitar camas de observación transitoria (UCE) y coordinar derivaciones vía UGCC.',
            'ciudadano': 'El hospital está, o estará, sobre su capacidad. Para atención no urgente, considera consultar otro centro si es posible.',
        }
