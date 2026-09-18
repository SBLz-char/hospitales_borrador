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

        # Motor de derivación: si el área queda sobre el umbral (85%), se ofrecen los
        # hospitales del mismo servicio de salud que para ese mismo mes proyectan menos.
        # Toda la lógica está en logica.sugerir_traslado(); acá solo se arma el listado.
        if resultado['valor'] > logica.UMBRAL_TRASLADO:
            with st.spinner('Buscando alternativas de derivación…'):
                sugerencia = logica.sugerir_traslado(
                    df_activo, resultado['establecimiento_cod'], resultado['area_cod'],
                    resultado['anio_obj'], resultado['mes_obj'],
                )
            st.markdown(f"**Alternativas de derivación · servicio {sugerencia['servicio']} · "
                        f"proyección {etiqueta_mes}**")
            if sugerencia['sin_alternativas']:
                st.info(f'Ningún otro hospital del servicio {sugerencia["servicio"]} proyecta menos de '
                        f'{logica.UMBRAL_TRASLADO:.0f}% para {etiqueta_mes}. Con la red sin holgura, '
                        'corresponde evaluar altas a domicilio de los pacientes que, según su nivel de '
                        'criticidad, puedan continuar su tratamiento en el hogar.')
            else:
                for hospital_alt in sugerencia['hospitales']:
                    etiqueta = ' · tiene la misma área' if hospital_alt['misma_area'] else ''
                    st.markdown(f"**{hospital_alt['nombre']}**{etiqueta}")
                    for area_alt in hospital_alt['areas']:
                        marca = ' · **misma área consultada**' if area_alt['misma_area'] else ''
                        st.markdown(f"- {area_alt['area']} — {area_alt['valor']:.1f}%{marca}")

        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown('**Para gestión hospitalaria**')
            st.write(semaforo['institucional'])
        with rc2:
            st.markdown('**Para la ciudadanía**')
            st.write(semaforo['ciudadano'])
