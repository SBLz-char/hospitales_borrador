"""
App de Streamlit: predicción de ocupación hospitalaria para cualquier hospital público
de la Red REM 20 (o uno nuevo que suba su propio historial). Alternativa gratuita a
Hugging Face Spaces, pensada para Streamlit Community Cloud.

Toda la lógica de negocio (carga del modelo, combinar lo subido con lo conocido, armar
la predicción) vive en logica.py, compartida con app.py (la versión de Gradio) -- acá
solo se traduce esa lógica a componentes de Streamlit.

Cómo se prueba localmente: `streamlit run streamlit_app.py`.
"""
import html
import re

import altair as alt  # viene incluido con Streamlit: no hay que agregarlo a requirements.txt
import pandas as pd
import streamlit as st

import logica

st.set_page_config(page_title='Predicción de Ocupación Hospitalaria', page_icon='🏥', layout='wide')

st.title('Predicción de Ocupación Hospitalaria')
st.caption('Modelo estandarizado — cualquier hospital público de la Red REM 20')

if 'df_subido' not in st.session_state:
    st.session_state.df_subido = None
    st.session_state.archivo_firma = None
    st.session_state.df_activo = None  # referencia + lo subido, con los nombres ya unificados

col_izq, col_der = st.columns([1, 1.3])

with col_izq:
    st.markdown('**1. Cargar datos (opcional)**')
    archivo = st.file_uploader('CSV de un hospital (mismo formato REM 20)', type=['csv'])

    if archivo is None:
        if st.session_state.archivo_firma is not None:
            st.session_state.df_activo = None  # se quitó el archivo: hay que rearmar
        st.session_state.df_subido = None
        st.session_state.archivo_firma = None
    else:
        firma = (archivo.name, archivo.size)
        if st.session_state.archivo_firma != firma:
            df_subido, mensaje, es_error = logica.cargar_archivo_subido(archivo)
            st.session_state.archivo_firma = firma
            st.session_state.df_subido = df_subido
            st.session_state.archivo_mensaje = mensaje
            st.session_state.archivo_es_error = es_error
            st.session_state.df_activo = None  # cambió el archivo: hay que rearmar
        (st.error if st.session_state.archivo_es_error else st.success)(st.session_state.archivo_mensaje)

    # df_activo se arma una sola vez por archivo y no en cada interacción: combinar la
    # referencia con un CSV grande y unificar nombres cuesta, y Streamlit re-ejecuta el
    # script completo cada vez que se toca un widget.
    if st.session_state.df_activo is None:
        st.session_state.df_activo = logica.dataframe_activo(st.session_state.df_subido)
    df_activo = st.session_state.df_activo

    st.markdown('**2. Elegir qué predecir**')
    texto_busqueda = st.text_input(
        'Buscar hospital',
        placeholder='ej. "hospital sotero del rio" (sin tildes, nombre incompleto)',
    )
    # Solo hospitales con datos del año vigente: los que dejaron de reportar antes (ej.
    # hospitales de campaña cerrados en 2020) quedan fuera del selector, porque predecir
    # desde su último mes real devolvería un mes que ya pasó.
    opciones_hospital = logica.buscar_hospital(texto_busqueda, st.session_state.df_subido, df_activo) \
        if texto_busqueda else logica.hospitales_disponibles(st.session_state.df_subido, df_activo)

    if not opciones_hospital:
        st.warning('Ningún hospital coincide con esa búsqueda.')
        hospital = None
    else:
        hospital = st.selectbox('Hospital', opciones_hospital)

    # Cascade hospital -> área: se ofrecen solo las áreas que ese hospital tiene datos
    # cargados, en vez de las 29 de la red (ej. Peumo tiene 4). La lógica de datos vive en
    # logica.areas_disponibles(); acá solo se consume. Como Streamlit re-ejecuta el script
    # al cambiar el selectbox de hospital, la lista de áreas se actualiza sola.
    cod_hospital = logica.codigo_de_hospital(df_activo, hospital) if hospital else None
    areas_hospital = logica.areas_disponibles(df_activo, cod_hospital)

    if areas_hospital:
        area = st.selectbox('Área funcional', areas_hospital)
        # Aviso de cobertura: logica.cobertura_area() dice desde y hasta qué mes hay datos
        # reales de esa área. Si no llega al último mes del dataset, se avisa, porque la
        # predicción parte desde ese último mes y no desde hoy.
        cobertura = logica.cobertura_area(df_activo, cod_hospital, logica._codigo_area(area, df_activo))
        if cobertura and not cobertura['al_dia']:
            st.warning(f'El área "{area}" de {hospital} tiene datos desde '
                       f'{cobertura["desde"]} hasta {cobertura["hasta"]}, y no hasta '
                       f'{cobertura["corte_datos"]} como el resto del dataset. '
                       'La predicción parte desde ese último mes con datos.')
    else:
        area = None
        if hospital:
            st.warning(f'No hay áreas con datos cargados para {hospital}.')

    horizonte = st.radio('Horizonte de predicción', list(logica.HORIZONTES.keys()), index=2, horizontal=True)
    predecir_click = st.button('Predecir ocupación', type='primary', use_container_width=True,
                               disabled=not (hospital and area))

if predecir_click:
    st.session_state.ultimo_resultado = logica.predecir_valor(
        hospital, area, horizonte, st.session_state.df_subido
    )

with col_der:
    resultado = st.session_state.get('ultimo_resultado')
    if not resultado:
        st.info('Elige un hospital, un área y un horizonte, y presiona "Predecir ocupación".')
    elif not resultado['ok']:
        (st.error if resultado['es_error'] else st.info)(resultado['mensaje'])
    else:
        semaforo = resultado['semaforo']
        tipo = 'proyección' if resultado['es_prediccion'] else 'dato real'
        etiqueta_mes = f"{resultado['mes_obj']:02d}/{resultado['anio_obj']}"

        st.markdown(f"### {semaforo['color']} — {semaforo['nivel']}")
        st.metric(label=f'Índice ocupacional · {tipo} · {etiqueta_mes}', value=f"{resultado['valor']:.1f}%")

        fig = logica.graficar_trayectoria(resultado['trayectoria'])
        st.pyplot(fig, use_container_width=True)


# ── Recomendaciones: sección de ancho completo, debajo del botón y del gráfico ─────

def _titulo_centrado(texto, nivel=2):
    st.markdown(f"<h{nivel} style='text-align:center'>{html.escape(texto)}</h{nivel}>",
                unsafe_allow_html=True)


# Colores del gráfico de alternativas (forma de 'énfasis'): el área recomendada en azul,
# la más desocupada en naranja y el resto en gris.
_COLORES_ALTERNATIVAS = {
    'dark': {'Recomendada': '#3987e5', 'Más baja': '#d95926', 'Otras': '#6b6a66', 'texto': '#fafafa'},
    'light': {'Recomendada': '#2a78d6', 'Más baja': '#eb6834', 'Otras': '#a3a29c', 'texto': '#31333f'},
}


def _etiqueta_area(nombre):
    """'Área de Hospitalización de ...' -> 'Hospitalización de ...': el prefijo se repite en
    todas y ocupa el ancho que necesita el nombre. El nombre completo va en el tooltip."""
    corto = re.sub(r'^Área (de )?', '', nombre).strip()
    return corto[:1].upper() + corto[1:]


def _grafico_alternativas(areas):
    """Barras horizontales de las áreas bajo el umbral de un hospital sugerido, de menor a
    mayor. Se destacan la recomendada (areas[0]: logica.sugerir_traslado la entrega primero,
    la misma área consultada o, si no la tiene, la más desocupada) y la más baja."""
    tema = getattr(getattr(st.context, 'theme', None), 'type', None) or 'dark'
    colores = _COLORES_ALTERNATIVAS.get(tema, _COLORES_ALTERNATIVAS['dark'])
    recomendada = areas[0]['area']
    minimo = min(a['valor'] for a in areas)

    filas = []
    for a in areas:
        # si la recomendada es además la más baja, gana 'Recomendada'
        grupo = ('Recomendada' if a['area'] == recomendada
                 else 'Más baja' if a['valor'] == minimo else 'Otras')
        filas.append({'Área': a['area'], 'Etiqueta': _etiqueta_area(a['area']),
                      'Índice (%)': round(a['valor'], 1), 'Grupo': grupo,
                      'Texto': f"{a['valor']:.1f}%"})
    datos = pd.DataFrame(filas)
    orden = datos.sort_values('Índice (%)')['Etiqueta'].tolist()
    grupos = [g for g in ('Recomendada', 'Más baja', 'Otras') if g in set(datos['Grupo'])]

    base = alt.Chart(datos).encode(
        y=alt.Y('Etiqueta:N', sort=orden, title=None,
                axis=alt.Axis(labelLimit=230, labelOverlap=False, ticks=False, domain=False)),
        tooltip=[alt.Tooltip('Área:N'), alt.Tooltip('Índice (%):Q', format='.1f'),
                 alt.Tooltip('Grupo:N', title='Destacada como')],
    )
    barras = base.mark_bar(cornerRadiusEnd=4, height={'band': 0.62}).encode(
        x=alt.X('Índice (%):Q', scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(values=[0, 20, 40, 60, 80, 100],
                              title=f'Índice ocupacional proyectado (%) · punteado: umbral {logica.UMBRAL_TRASLADO:.0f}%')),
        color=alt.Color('Grupo:N', title=None,
                        scale=alt.Scale(domain=grupos, range=[colores[g] for g in grupos]),
                        legend=alt.Legend(orient='bottom', direction='horizontal')),
    )
    # todas las áreas listadas están bajo 85%, así que el valor cabe a la derecha de la barra
    valores = base.mark_text(align='left', dx=4, color=colores['texto']).encode(
        x='Índice (%):Q', text='Texto:N')
    linea = alt.Chart(pd.DataFrame({'x': [logica.UMBRAL_TRASLADO]})).mark_rule(
        strokeDash=[4, 4], opacity=0.7, color=colores['texto']).encode(x='x:Q')
    # alto por fila (step) y no total: así cada área tiene su espacio aunque sean muchas,
    # y los ejes y la leyenda se suman aparte en vez de comerse el alto de las barras
    return (barras + valores + linea).properties(height=alt.Step(38))


resultado = st.session_state.get('ultimo_resultado')
if resultado and resultado['ok']:
    semaforo = resultado['semaforo']
    etiqueta_mes = f"{resultado['mes_obj']:02d}/{resultado['anio_obj']}"

    st.divider()
    _titulo_centrado('Recomendaciones a seguir')
    # una debajo de la otra, no en dos columnas: se leen como una lista
    st.subheader('Para gestión hospitalaria')
    st.markdown(f"- {semaforo['institucional']}")
    st.subheader('Para la ciudadanía')
    st.markdown(f"- {semaforo['ciudadano']}")

    # Motor de derivación: si el área queda sobre el umbral (85%), se comparan los 2
    # hospitales del mismo servicio de salud que para ese mismo mes proyectan menos.
    # Toda la lógica está en logica.sugerir_traslado(); acá solo se presenta.
    if resultado['valor'] > logica.UMBRAL_TRASLADO:
        _titulo_centrado('Sugerencia de hospitales para traslado')
        with st.spinner('Buscando alternativas de derivación…'):
            sugerencia = logica.sugerir_traslado(
                df_activo, resultado['establecimiento_cod'], resultado['area_cod'],
                resultado['anio_obj'], resultado['mes_obj'], maximo=2,
            )
        st.markdown(f"<p style='text-align:center'>Servicio de salud {html.escape(str(sugerencia['servicio']))} "
                    f"· proyección {etiqueta_mes} · áreas bajo {logica.UMBRAL_TRASLADO:.0f}%</p>",
                    unsafe_allow_html=True)

        if sugerencia['sin_alternativas']:
            st.info(f'Ningún otro hospital del servicio {sugerencia["servicio"]} proyecta menos de '
                    f'{logica.UMBRAL_TRASLADO:.0f}% para {etiqueta_mes}. Con la red sin holgura, '
                    'corresponde evaluar altas a domicilio de los pacientes que, según su nivel de '
                    'criticidad, puedan continuar su tratamiento en el hogar.')
        else:
            # un hospital debajo del otro, cada uno a todo el ancho
            for posicion, hospital_alt in enumerate(sugerencia['hospitales']):
                if posicion:
                    st.divider()
                recomendada = hospital_alt['areas'][0]
                mas_baja = min(hospital_alt['areas'], key=lambda a: a['valor'])
                _titulo_centrado(hospital_alt['nombre'], nivel=3)
                st.markdown(f"**Área recomendada:** {recomendada['area']} — {recomendada['valor']:.1f}%")
                if mas_baja['area'] != recomendada['area']:
                    st.markdown(f"**Más desocupada:** {mas_baja['area']} — {mas_baja['valor']:.1f}%")
                st.altair_chart(_grafico_alternativas(hospital_alt['areas']),
                                use_container_width=True, theme='streamlit')
                with st.expander('Ver valores en tabla'):
                    st.dataframe(
                        pd.DataFrame([{'Área': a['area'], 'Índice proyectado (%)': round(a['valor'], 1)}
                                      for a in hospital_alt['areas']]),
                        hide_index=True, use_container_width=True,
                    )
