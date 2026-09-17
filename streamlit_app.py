"""
App de Streamlit: predicción de ocupación hospitalaria para cualquier hospital público
de la Red REM 20 (o uno nuevo que suba su propio historial). Alternativa gratuita a
Hugging Face Spaces, pensada para Streamlit Community Cloud.

Toda la lógica de negocio (carga del modelo, combinar lo subido con lo conocido, armar
la predicción) vive en logica.py, compartida con app.py (la versión de Gradio) -- acá
solo se traduce esa lógica a componentes de Streamlit.

Cómo se prueba localmente: `streamlit run streamlit_app.py`.
"""
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
    opciones_hospital = logica.buscar_hospital(texto_busqueda, st.session_state.df_subido) \
        if texto_busqueda else logica.hospitales_disponibles(st.session_state.df_subido)

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
    st.session_state.ultimo_horizonte = horizonte  # para rotular la métrica del resultado

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

        # Precisión del modelo: logica.texto_metricas(etiqueta) da la cifra del horizonte
        # pedido, o la global si se llama sin argumento. Devuelve '' si el .pkl no trae
        # métricas, así que no hace falta un try/except.
        texto_precision = logica.texto_metricas(st.session_state.get('ultimo_horizonte'))
        if texto_precision:
            st.caption(f'{texto_precision}. Es el error promedio del modelo en la validación '
                       'histórica, no la precisión de esta predicción en particular.')

        fig = logica.graficar_trayectoria(resultado['trayectoria'])
        st.pyplot(fig, use_container_width=True)

        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown('**Para gestión hospitalaria**')
            st.write(semaforo['institucional'])
        with rc2:
            st.markdown('**Para la ciudadanía**')
            st.write(semaforo['ciudadano'])
