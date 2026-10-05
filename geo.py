# -*- coding: utf-8 -*-
"""
Capa geográfica: región, comuna y distancia entre hospitales.

Los datos vienen de hospitales_geo.parquet, una tabla de apoyo de 192 filas construida con
distancias/construir_hospitales_geo.py a partir de "Establecimientos de Salud vigentes"
(DEIS / Ministerio de Salud, datos.gob.cl, licencia CC0). El extracto del REM 20
(dataset_referencia.parquet) NO se modifica: esta tabla se cruza por
CODIGO_ESTABLECIMIENTO solo en el momento de recomendar, igual que una tabla temporal.

Por qué hace falta: hasta ahora las alternativas de traslado se buscaban dentro del mismo
servicio de salud, asumiendo que eso equivalía a la misma región. Vale para 28 de los 29
servicios. La excepción es el Hospital Hanga Roa, que depende del Servicio Metropolitano
Oriente pero está en Isla de Pascua, a 3.768 km de Santiago: la app llegó a ofrecerlo como
destino de traslado para el Hospital Del Salvador. Con esta tabla el filtro deja de ser
"mismo servicio" y pasa a ser "mismo servicio Y misma región", con la región puesta por el
Ministerio y no por una lista escrita a mano.

Todo el módulo está escrito para no romper la app si el .parquet no está: en ese caso
DISPONIBLE queda en False, cada función devuelve None y logica.py sigue funcionando con el
comportamiento anterior (filtrar solo por servicio de salud).
"""
import math
import os

import pandas as pd

DIR_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_GEO = os.path.join(DIR_BASE, 'hospitales_geo.parquet')

RADIO_TIERRA_KM = 6371.0

try:
    _GEO = pd.read_parquet(RUTA_GEO)
    _GEO['CODIGO_ESTABLECIMIENTO'] = _GEO['CODIGO_ESTABLECIMIENTO'].astype('int64')
    _POR_CODIGO = _GEO.set_index('CODIGO_ESTABLECIMIENTO')
    REGION_POR_CODIGO = _POR_CODIGO['REGION'].to_dict()
    COMUNA_POR_CODIGO = _POR_CODIGO['COMUNA'].to_dict()
    _COORDENADAS = {cod: (lat, lon) for cod, lat, lon
                    in zip(_GEO['CODIGO_ESTABLECIMIENTO'], _GEO['LATITUD'], _GEO['LONGITUD'])}
    DISPONIBLE = True
    MOTIVO_NO_DISPONIBLE = ''
except Exception as e:  # falta el archivo, está corrupto, o cambió de formato
    _GEO = None
    REGION_POR_CODIGO = {}
    COMUNA_POR_CODIGO = {}
    _COORDENADAS = {}
    DISPONIBLE = False
    MOTIVO_NO_DISPONIBLE = f'{type(e).__name__}: {e}'


def region_de(establecimiento_cod):
    """Región del hospital según el DEIS, o None si no está en la tabla."""
    if establecimiento_cod is None:
        return None
    return REGION_POR_CODIGO.get(int(establecimiento_cod))


def comuna_de(establecimiento_cod):
    """Comuna del hospital según el DEIS, o None si no está en la tabla."""
    if establecimiento_cod is None:
        return None
    return COMUNA_POR_CODIGO.get(int(establecimiento_cod))


def coordenadas_de(establecimiento_cod):
    """(latitud, longitud) del hospital, o None."""
    if establecimiento_cod is None:
        return None
    return _COORDENADAS.get(int(establecimiento_cod))


def misma_region(cod_a, cod_b):
    """True si los dos hospitales están en la misma región.

    Si de alguno no se conoce la región, devuelve True en vez de False: sin dato, se
    prefiere no esconder una alternativa que podría ser válida -- el filtro por servicio de
    salud, que sigue aplicándose antes, ya acota la búsqueda."""
    region_a, region_b = region_de(cod_a), region_de(cod_b)
    if region_a is None or region_b is None:
        return True
    return region_a == region_b


def distancia_km(cod_a, cod_b):
    """Distancia en línea recta entre dos hospitales, en kilómetros, o None si falta la
    coordenada de alguno.

    Es distancia geodésica (la "en línea recta" sobre la superficie de la Tierra), no la
    distancia por carretera: siempre es menor o igual a la real. Sirve para ordenar
    alternativas de más cerca a más lejos, que es para lo que se usa; no para estimar un
    tiempo de viaje."""
    a, b = coordenadas_de(cod_a), coordenadas_de(cod_b)
    if a is None or b is None:
        return None
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * RADIO_TIERRA_KM * math.asin(math.sqrt(h))


def texto_distancia(km):
    """Distancia lista para mostrar. Bajo 10 km se usa un decimal, porque entre hospitales
    de una misma ciudad la diferencia entre 2,7 y 8,3 km importa; sobre 10 km el decimal es
    ruido frente al error de usar línea recta en vez de ruta."""
    if km is None:
        return ''
    if km < 1:
        return f'{km * 1000:.0f} m'
    if km < 10:
        return f'{km:.1f} km'.replace('.', ',')
    return f'{km:,.0f} km'.replace(',', '.')
