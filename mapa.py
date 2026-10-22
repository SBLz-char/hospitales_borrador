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
"""
import math

import pydeck as pdk

import geo

# rojo para el hospital consultado (el que esta saturado), verde para las alternativas:
# el mismo criterio de color que ya usan las insignias de la app
COLOR_ORIGEN = [208, 59, 59]
COLOR_DESTINO = [12, 163, 12]
COLOR_LINEA = [120, 120, 120]

# El texto va sobre el mapa, que es claro u oscuro segun el tema del usuario. Sin esto las
# etiquetas quedan ilegibles: gris sobre gris en modo oscuro.
TEXTO_CLARO = {'color': [45, 45, 45], 'fondo': [255, 255, 255, 205]}
TEXTO_OSCURO = {'color': [240, 240, 240], 'fondo': [20, 20, 20, 205]}

RADIO_ORIGEN = 220
RADIO_DESTINO = 160
ALTO_MAPA = 420


def _zoom_para(latitudes, longitudes):
    """Nivel de zoom que deja todos los puntos a la vista.

    Se calcula a partir del lado mas grande del rectangulo que los contiene. El factor 1,6
    deja un margen alrededor para que ningun marcador quede pegado al borde."""
    span_lat = max(latitudes) - min(latitudes)
    span_lon = (max(longitudes) - min(longitudes)) * math.cos(math.radians(sum(latitudes) / len(latitudes)))
    span = max(span_lat, abs(span_lon), 0.01)
    return max(4.0, min(13.0, math.log2(360.0 / (span * 1.6))))


def construir_mapa(establecimiento_cod, nombre_origen, valor_origen, sugerencia,
                   modo_oscuro=False):
    """Devuelve un pydeck.Deck con el hospital consultado y sus alternativas, o None.

    establecimiento_cod -- codigo del hospital al que se le hizo la prediccion
    nombre_origen       -- nombre para la etiqueta y el tooltip
    valor_origen        -- indice proyectado del hospital consultado, para el tooltip
    sugerencia          -- lo que devuelve logica.sugerir_traslado()
    modo_oscuro         -- True si la app esta en tema oscuro, para el color de las etiquetas
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
        'etiqueta': 'Consultado',
        'color': COLOR_ORIGEN, 'radio': RADIO_ORIGEN,
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
            'etiqueta': f"{puesto}. {alt.get('distancia_texto') or ''}".strip(),
            'color': COLOR_DESTINO, 'radio': RADIO_DESTINO,
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

    tema = TEXTO_OSCURO if modo_oscuro else TEXTO_CLARO
    capas = [
        # la linea recta entre los dos hospitales: es lo que mide la distancia que se muestra
        pdk.Layer('LineLayer', data=lineas, get_source_position='origen',
                  get_target_position='destino', get_color=COLOR_LINEA, get_width=2,
                  pickable=False),
        pdk.Layer('ScatterplotLayer', data=puntos, get_position=['lon', 'lat'],
                  get_fill_color='color', get_radius='radio', radius_min_pixels=7,
                  radius_max_pixels=24, stroked=True, get_line_color=[255, 255, 255],
                  line_width_min_pixels=2, pickable=True),
        # la distancia escrita junto a cada marcador, para que se lea sin pasar el mouse
        pdk.Layer('TextLayer', data=puntos, get_position=['lon', 'lat'], get_text='etiqueta',
                  get_size=13, get_color=tema['color'], get_pixel_offset=[0, -22],
                  get_text_anchor='"middle"', get_alignment_baseline='"bottom"',
                  background=True, get_background_color=tema['fondo'],
                  background_padding=[5, 3, 5, 3], pickable=False),
    ]

    return pdk.Deck(
        layers=capas,
        initial_view_state=vista,
        # scrollZoom apagado: con la rueda activada, bajar la pagina con el cursor encima
        # del mapa hace zoom en vez de desplazar, y el usuario queda atrapado. Se puede
        # seguir arrastrando el mapa y usando los botones + y - de la esquina.
        views=[pdk.View(type='MapView', controller={'scrollZoom': False})],
        # map_style=None deja que Streamlit elija el mapa claro u oscuro segun el tema que
        # tenga puesto el usuario. No se usa Mapbox, asi que no hace falta ninguna clave.
        map_style=None,
        tooltip={'html': '<b>{nombre}</b><br/>{comuna}<br/>{detalle}',
                 'style': {'fontSize': '12px'}},
        height=ALTO_MAPA,
    )
