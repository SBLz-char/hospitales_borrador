# -*- coding: utf-8 -*-
"""
Mapa de la sugerencia de traslado: el hospital consultado, las alternativas propuestas y
la distancia entre ellos.

No usa ningun servicio pago ni clave de API. Las coordenadas salen de hospitales_geo.parquet
(tabla propia, construida con datos del DEIS) y los mosaicos del mapa los sirve Carto, que es
el proveedor que Streamlit trae por omision para st.pydeck_chart. pydeck viene como
dependencia obligatoria de Streamlit, asi que requirements.txt no cambia.

Si falta la coordenada de alguno, construir_mapa() devuelve None y la app simplemente no
muestra el mapa: la lista de alternativas de texto sigue estando igual.

Cada punto lleva su nombre: el consultado, debajo del punto rojo; cada alternativa, encima
de su punto verde junto con la distancia. Las etiquetas son texto blanco sobre un recuadro
oscuro semitransparente, que se lee igual sobre el mapa claro y sobre el oscuro, así que no
dependen del tema de la app.
"""
import math

import pydeck as pdk

import geo

# rojo para el hospital consultado (el que esta saturado), verde para las alternativas:
# el mismo criterio de color que ya usan las insignias de la app
COLOR_ORIGEN = [208, 59, 59]
COLOR_DESTINO = [12, 163, 12]
COLOR_LINEA = [120, 120, 120]

# Etiquetas: el mismo estilo en tema claro y oscuro (ver el docstring del modulo).
COLOR_TEXTO = [255, 255, 255]
COLOR_FONDO_TEXTO = [25, 28, 34, 215]
TAMANO_TEXTO = 13
# ancho maximo de una linea de etiqueta, en multiplos del tamano de letra: los nombres del
# REM son largos ("Complejo Hospitalario Dr. Sotero del Rio (Santiago, Puente Alto)") y sin
# esto cruzarian todo el mapa
ANCHO_ETIQUETA = 20

RADIO_ORIGEN = 220
RADIO_DESTINO = 160
ALTO_MAPA = 420


def _caracteres(puntos):
    """Caracteres que necesitan las etiquetas: el ASCII imprimible mas lo que traigan los
    nombres (tildes, ñ, el punto medio de "nombre · distancia").

    Se devuelve como un solo texto entre comillas dobles. pydeck convierte cualquier texto
    (y cualquier lista de textos) en una expresion de JavaScript, salvo que venga entre
    comillas: en ese caso se las quita y lo pasa tal cual, y deck.gl acepta un texto como
    conjunto de caracteres. Las comillas dobles se sacan del conjunto para no cerrar el
    texto antes de tiempo; ningun nombre de hospital las usa."""
    ascii_imprimible = {chr(c) for c in range(32, 127)}
    caracteres = ascii_imprimible | set(''.join(p['etiqueta'] for p in puntos))
    caracteres.discard('"')
    return '"' + ''.join(sorted(caracteres)) + '"'


def _zoom_para(latitudes, longitudes):
    """Nivel de zoom que deja todos los puntos a la vista.

    Se calcula a partir del lado mas grande del rectangulo que los contiene. El factor 2,4
    deja un margen alrededor para que ni los marcadores ni sus nombres, que van encima o
    debajo de cada punto, queden cortados contra el borde.

    El piso de 0,002 grados (unos 200 m) y el tope de zoom 16 son para hospitales muy
    cercanos entre si: Del Salvador, Asenjo y el INER estan a menos de 300 m, y con menos zoom
    sus nombres quedarian encimados."""
    span_lat = max(latitudes) - min(latitudes)
    span_lon = (max(longitudes) - min(longitudes)) * math.cos(math.radians(sum(latitudes) / len(latitudes)))
    span = max(span_lat, abs(span_lon), 0.002)
    return max(4.0, min(16.0, math.log2(360.0 / (span * 2.4))))


def construir_mapa(establecimiento_cod, nombre_origen, valor_origen, sugerencia):
    """Devuelve un pydeck.Deck con el hospital consultado y sus alternativas, o None.

    establecimiento_cod -- codigo del hospital al que se le hizo la prediccion
    nombre_origen       -- nombre para la etiqueta y el tooltip
    valor_origen        -- indice proyectado del hospital consultado, para el tooltip
    sugerencia          -- lo que devuelve logica.sugerir_traslado()
    """
    if not geo.DISPONIBLE or not sugerencia:
        return None
    origen = geo.coordenadas_de(establecimiento_cod)
    if origen is None:
        return None

    puntos = [{
        'lat': origen[0], 'lon': origen[1], 'nombre': nombre_origen,
        'comuna': geo.comuna_de(establecimiento_cod) or '',
        'detalle': f'Hospital consultado · {valor_origen:.1f}% proyectado',
        'etiqueta': nombre_origen,
        'color': COLOR_ORIGEN, 'radio': RADIO_ORIGEN,
        'desplazamiento': [0, 20], 'base': 'top',
    }]
    lineas = []

    for puesto, alt in enumerate(sugerencia.get('hospitales', []), start=1):
        destino = geo.coordenadas_de(alt['codigo'])
        if destino is None:
            continue
        # misma area recomendada que muestra la lista de abajo: areas[0], que logica.py
        # ya dejo ordenada con el criterio clinico primero
        recomendada = (alt.get('areas') or [{}])[0]
        detalle = f'Alternativa {puesto}'
        if recomendada.get('area'):
            detalle += f" · {recomendada['area']}"
        if recomendada.get('valor') is not None:
            detalle += f" · {recomendada['valor']:.1f}%"
        puntos.append({
            'lat': destino[0], 'lon': destino[1], 'nombre': alt['nombre'],
            'comuna': alt.get('comuna') or '',
            'detalle': detalle,
            'etiqueta': (f"{alt['nombre']} · {alt['distancia_texto']}"
                         if alt.get('distancia_texto') else alt['nombre']),
            'color': COLOR_DESTINO, 'radio': RADIO_DESTINO,
            'desplazamiento': [0, -20], 'base': 'bottom',
        })
        lineas.append({
            'origen': [origen[1], origen[0]], 'destino': [destino[1], destino[0]],
            'nombre': alt['nombre'], 'distancia': alt.get('distancia_texto') or '',
        })

    if not lineas:
        return None

    latitudes = [p['lat'] for p in puntos]
    longitudes = [p['lon'] for p in puntos]
    vista = pdk.ViewState(
        latitude=sum(latitudes) / len(latitudes),
        longitude=sum(longitudes) / len(longitudes),
        zoom=_zoom_para(latitudes, longitudes),
    )

    capas = [
        # la linea recta entre los dos hospitales: es lo que mide la distancia que se muestra
        pdk.Layer('LineLayer', data=lineas, get_source_position='origen',
                  get_target_position='destino', get_color=COLOR_LINEA, get_width=2,
                  pickable=False),
        pdk.Layer('ScatterplotLayer', data=puntos, get_position=['lon', 'lat'],
                  get_fill_color='color', get_radius='radio', radius_min_pixels=7,
                  radius_max_pixels=24, stroked=True, get_line_color=[255, 255, 255],
                  line_width_min_pixels=2, pickable=True),
        # el nombre de cada hospital junto a su punto (y la distancia, en las alternativas),
        # para que se lea sin pasar el mouse. character_set lleva la lista de caracteres a
        # dibujar: por omision deck.gl solo trae ASCII y las tildes y la ñ no aparecerian.
        # Los valores de texto van entre comillas dobles por la regla de pydeck explicada
        # en _caracteres(): sin ellas se convierten en expresiones y la capa no se dibuja.
        pdk.Layer('TextLayer', data=puntos, get_position=['lon', 'lat'], get_text='etiqueta',
                  get_size=TAMANO_TEXTO, get_color=COLOR_TEXTO, get_pixel_offset='desplazamiento',
                  get_text_anchor='"middle"', get_alignment_baseline='base',
                  character_set=_caracteres(puntos), font_family='"Helvetica, Arial, sans-serif"',
                  word_break='"break-word"', max_width=ANCHO_ETIQUETA,
                  background=True, get_background_color=COLOR_FONDO_TEXTO,
                  background_padding=[6, 4, 6, 4], pickable=False),
    ]

    return pdk.Deck(
        layers=capas,
        initial_view_state=vista,
        # zoom con la rueda del mouse activado, ademas de arrastrar y de los botones + y -
        views=[pdk.View(type='MapView', controller=True)],
        # map_style=None deja que Streamlit elija el mapa claro u oscuro segun el tema que
        # tenga puesto el usuario. No se usa Mapbox, asi que no hace falta ninguna clave.
        map_style=None,
        tooltip={'html': '<b>{nombre}</b><br/>{comuna}<br/>{detalle}',
                 'style': {'fontSize': '12px'}},
        height=ALTO_MAPA,
    )
