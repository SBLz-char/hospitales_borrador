"""
App de Streamlit: predicción de ocupación hospitalaria para cualquier hospital público
de la Red REM 20 (o uno nuevo que suba su propio historial). Alternativa gratuita a
Hugging Face Spaces, pensada para Streamlit Community Cloud.

Toda la lógica de negocio (carga del modelo, combinar lo subido con lo conocido, armar
la predicción) vive en logica.py, compartida con app.py (la versión de Gradio) -- acá
solo se traduce esa lógica a componentes de Streamlit.

Estilo visual: tarjetas con st.container(border=True) + una hoja de estilos chica
inyectada abajo (insignia de estado, número hero, pastillas de fecha/margen de error),
con soporte de modo oscuro (ver _ES_OSCURO más abajo, y [theme.light]/[theme.dark] en
.streamlit/config.toml). Es solo piel -- ningún cambio acá toca
logica.py/carga_datos.py/inferencia.py, así que el resultado de cada predicción es
exactamente el mismo que antes.

Cómo se prueba localmente: `streamlit run streamlit_app.py`.
"""
import html
import re

import altair as alt  # viene incluido con Streamlit: no hay que agregarlo a requirements.txt
import pandas as pd
import streamlit as st

import explicacion
import logica

st.set_page_config(page_title='Predicción de Ocupación Hospitalaria', page_icon='🏥', layout='wide')

# ── Modo oscuro ─────────────────────────────────────────────────────────────────────
# st.context.theme.type refleja el tema REALMENTE aplicado (System/Light/Dark, lo que el
# usuario haya elegido en el menú ⋮ de Streamlit), a diferencia de @media
# (prefers-color-scheme) que solo ve la preferencia del sistema operativo -- por eso el
# color de cada elemento propio (badges, tarjetas, gráficos) se decide acá, en Python, en
# vez de con una media query en el CSS. Ojo: este valor se actualiza recién en el próximo
# rerun después de cambiar el tema (ej. al tocar cualquier filtro), no de forma instantánea
# -- una limitación conocida de Streamlit, no de esta app.
_ES_OSCURO = st.context.theme.type == 'dark'

# ── Estilo (Estilo 1 "clínico moderno" del mockup) ─────────────────────────────────
# Solo CSS -- ningún selector de acá cambia el valor que devuelve un widget, así que si
# una regla no calza con una versión futura de Streamlit, en el peor caso se pierde el
# detalle visual (ej. el color del botón), nunca la funcionalidad. Los tokens de color se
# arman abajo según _ES_OSCURO en vez de venir fijos, para que las tarjetas/badges propios
# no queden con fondo claro sobre un Streamlit en modo oscuro (o viceversa).
if _ES_OSCURO:
    _T = dict(
        badge_good='#0ca30c', badge_neutral='#c3c2b7', badge_warning='#fab219', badge_critical='#d03b3b',
        badge_good_bg='rgba(12,163,12,.18)', badge_neutral_bg='rgba(255,255,255,.08)',
        badge_warning_bg='rgba(250,178,25,.18)', badge_critical_bg='rgba(208,59,59,.22)',
        hero_num='#fafafa', hero_small='#9c9a94', hero_label='#948f89',
        pill_bg='rgba(255,255,255,.06)', pill_border='rgba(255,255,255,.16)', pill_fg='#c3c2b7',
        margin_bg='rgba(255,255,255,.08)', margin_fg='#c3c2b7',
        reco_inst_bg='#152438', reco_inst_icon='#1c3352', reco_cit_bg='#211a35', reco_cit_icon='#2f2650',
        reco_text='#d6d4cf',
        explica_bg='#17211c', explica_borde='#2f7d5b', explica_titulo='#6fbf9a',
    )
else:
    _T = dict(
        badge_good='#0ca30c', badge_neutral='#45586b', badge_warning='#a3690a', badge_critical='#c2453f',
        badge_good_bg='#eaf7ea', badge_neutral_bg='#eef1f5', badge_warning_bg='#fdf3e0', badge_critical_bg='#fbeae7',
        hero_num='#0b0b0b', hero_small='#6b6a66', hero_label='#898781',
        pill_bg='#ffffff', pill_border='#e1e0d9', pill_fg='#52514e',
        margin_bg='#f0efec', margin_fg='#52514e',
        reco_inst_bg='#eaf1fc', reco_inst_icon='#d7e6fa', reco_cit_bg='#f1edfc', reco_cit_icon='#e2d8f8',
        reco_text='#2f333d',
        explica_bg='#eef6f1', explica_borde='#2f7d5b', explica_titulo='#2a6a4d',
    )
# los "dot" de cada badge (VERDE/AZUL/AMARILLO/ROJO) usan siempre el tono de estado puro,
# tanto en claro como en oscuro -- solo el color del TEXTO de la insignia se atenúa en
# modo claro para que no quede demasiado saturado sobre un fondo pastel
_DOT = {'good': '#0ca30c', 'neutral': _T['badge_neutral'] if _ES_OSCURO else '#45586b',
        'warning': '#eda100', 'critical': '#d03b3b'}

st.markdown(f"""
<style>
.badge {{ display:inline-flex; align-items:center; gap:8px; font-size:14px; font-weight:600;
         padding:6px 14px; border-radius:999px; margin-bottom:10px; }}
.badge .dot {{ width:10px; height:10px; border-radius:50%; flex-shrink:0; }}
.badge-good {{ background:{_T['badge_good_bg']}; color:{_T['badge_good']}; }} .badge-good .dot {{ background:{_DOT['good']}; }}
.badge-neutral {{ background:{_T['badge_neutral_bg']}; color:{_T['badge_neutral']}; }} .badge-neutral .dot {{ background:{_DOT['neutral']}; }}
.badge-warning {{ background:{_T['badge_warning_bg']}; color:{_T['badge_warning']}; }} .badge-warning .dot {{ background:{_DOT['warning']}; }}
.badge-critical {{ background:{_T['badge_critical_bg']}; color:{_T['badge_critical']}; }} .badge-critical .dot {{ background:{_DOT['critical']}; }}
.hero-num {{ font-size:44px; font-weight:700; line-height:1; color:{_T['hero_num']}; }}
.hero-num small {{ font-size:20px; font-weight:600; color:{_T['hero_small']}; margin-left:2px; }}
.hero-label {{ font-size:12.5px; color:{_T['hero_label']}; margin:2px 0 10px; }}
.pill-info {{ display:inline-block; font-size:12px; color:{_T['pill_fg']}; background:{_T['pill_bg']};
             border:1px solid {_T['pill_border']}; padding:6px 13px; border-radius:999px; }}
.margin-pill {{ display:inline-block; font-size:12px; color:{_T['margin_fg']}; background:{_T['margin_bg']};
               padding:5px 11px; border-radius:8px; margin:2px 0 8px; }}
.reco-card {{ display:flex; gap:14px; align-items:center; min-height:74px; padding:18px 20px;
             border-radius:14px; }}
.reco-card.role-inst {{ background:{_T['reco_inst_bg']}; }}
.reco-card.role-cit {{ background:{_T['reco_cit_bg']}; }}
.reco-icon {{ width:46px; height:46px; min-width:46px; border-radius:13px;
             display:flex; align-items:center; justify-content:center; font-size:21px; }}
.reco-icon.role-inst {{ background:{_T['reco_inst_icon']}; }}
.reco-icon.role-cit {{ background:{_T['reco_cit_icon']}; }}
.reco-card p {{ font-size:14px; line-height:1.45; color:{_T['reco_text']}; margin:4px 0 0; }}
.reco-card b {{ font-size:14.5px; letter-spacing:.2px; }}
.caja-explica {{ background:{_T['explica_bg']}; border-left:4px solid {_T['explica_borde']};
                 border-radius:10px; padding:14px 16px; margin-top:4px; }}
.caja-explica p {{ font-size:14px; line-height:1.55; color:{_T['reco_text']}; margin:0; }}
.caja-explica .titulo {{ font-size:12px; font-weight:700; letter-spacing:.4px;
                        text-transform:uppercase; color:{_T['explica_titulo']}; margin-bottom:6px; }}
.camas-linea {{ display:inline-block; font-size:13px; color:{_T['pill_fg']}; background:{_T['margin_bg']};
               padding:6px 12px; border-radius:8px; margin:2px 0 8px; }}
h1#predicci-n-de-ocupaci-n-hospitalaria, .stApp h1 {{ font-size: 26px !important; }}
</style>
""", unsafe_allow_html=True)
# El color de acento (botón primario, st.pills seleccionado, foco de inputs) lo fija
# .streamlit/config.toml con [theme.light]/[theme.dark] (primaryColor en cada uno), no CSS
# a mano -- así Streamlit lo aplica solo, de forma consistente, en todos sus widgets
# nativos, y el usuario conserva el selector System/Light/Dark del menú ⋮.

# La clave de la API para la explicación con IA se lee de los Secrets de Streamlit
# (Manage app -> Settings -> Secrets) y se le pasa a explicacion.py, que es el único que la
# usa. Nunca va en el código ni en el repositorio. Si no está configurada, explicacion.py
# usa su texto determinista y la app funciona igual -- st.secrets lanza excepción cuando no
# hay archivo de secretos, así que la lectura va protegida.
try:
    explicacion.configurar_clave(st.secrets.get('ANTHROPIC_API_KEY'))
except Exception:
    explicacion.configurar_clave(None)

st.title('🏥 Predicción de Ocupación Hospitalaria')
st.caption('Modelo estandarizado — cualquier hospital público de la Red REM 20')

if 'df_subido' not in st.session_state:
    st.session_state.df_subido = None
    st.session_state.archivo_firma = None
    st.session_state.df_activo = None  # referencia + lo subido, con los nombres ya unificados
    st.session_state.archivo_detalle_tecnico = ''

col_izq, col_der = st.columns([1, 1.3])

with col_izq, st.container(border=True):
    st.markdown('**1. Cargar datos (opcional)**')
    archivo = st.file_uploader('CSV de un hospital (mismo formato REM 20)', type=['csv'])

    if archivo is None:
        if st.session_state.archivo_firma is not None:
            st.session_state.df_activo = None  # se quitó el archivo: hay que rearmar
        st.session_state.df_subido = None
        st.session_state.archivo_firma = None
        st.session_state.archivo_detalle_tecnico = ''
    else:
        firma = (archivo.name, archivo.size)
        if st.session_state.archivo_firma != firma:
            # cargar_archivo_subido_detallado() en vez de la forma corta: además del mensaje
            # amigable (Punto 3), trae 'detalle_tecnico' para el expander opcional de abajo —
            # útil para nosotros o para quien suba el archivo y no entienda por qué falló.
            resultado_carga = logica.cargar_archivo_subido_detallado(archivo)
            st.session_state.archivo_firma = firma
            st.session_state.df_subido = resultado_carga['df']
            st.session_state.archivo_mensaje = resultado_carga['mensaje']
            st.session_state.archivo_es_error = resultado_carga['es_error']
            st.session_state.archivo_detalle_tecnico = resultado_carga['detalle_tecnico']
            st.session_state.df_activo = None  # cambió el archivo: hay que rearmar
        (st.error if st.session_state.archivo_es_error else st.success)(st.session_state.archivo_mensaje)
        # Punto 3 (parte opcional): el mensaje de arriba nunca trae jerga ni nombres de
        # columnas, pero quien sepa leerlo puede querer el detalle exacto del fallo -- va
        # aparte, colapsado, para no ensuciar la vista de quien no lo necesita.
        if st.session_state.archivo_es_error and st.session_state.archivo_detalle_tecnico:
            with st.expander('Detalle técnico del error'):
                st.code(st.session_state.archivo_detalle_tecnico)

    # df_activo se arma una sola vez por archivo y no en cada interacción: combinar la
    # referencia con un CSV grande y unificar nombres cuesta, y Streamlit re-ejecuta el
    # script completo cada vez que se toca un widget.
    if st.session_state.df_activo is None:
        st.session_state.df_activo = logica.dataframe_activo(st.session_state.df_subido)
    df_activo = st.session_state.df_activo

    # Punto 6 de las anotaciones: antes no había forma de saber, sin buscarlo en el código,
    # hasta qué mes llegan los datos. logica.ultimo_periodo() ya existe para esto (la usa
    # cobertura_area() más abajo) y se calcula sobre df_activo -- que es referencia + lo
    # subido -- así que si el archivo cargado trae un mes más nuevo que la referencia, esta
    # fecha avanza sola sin tocar código. No se llama "datos de referencia" a secas porque,
    # con un archivo cargado, la fecha ya no describe solo la referencia sino la combinación:
    # el texto lo deja explícito para que no parezca un valor fijo.
    _anio_corte, _mes_corte = logica.ultimo_periodo(df_activo)
    _fecha_corte = f'{_mes_corte:02d}/{_anio_corte}'
    if st.session_state.df_subido is not None and not st.session_state.df_subido.empty:
        st.markdown(f'<span class="pill-info">📅 Datos disponibles hasta {_fecha_corte} '
                    f'(incluye tu archivo cargado)</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="pill-info">📅 Datos de referencia disponibles hasta '
                    f'{_fecha_corte}</span>', unsafe_allow_html=True)

    st.markdown('**2. Elegir qué predecir**')

    # Punto 4 de las anotaciones: si el archivo subido no corresponde, no se ofrecen
    # hospitales ni áreas de referencia como si nada hubiera pasado -- se bloquea la
    # selección hasta que el usuario corrija el archivo o lo quite.
    archivo_con_error = archivo is not None and st.session_state.get('archivo_es_error', False)

    if archivo_con_error:
        st.warning(
            'Corrige el archivo o quítalo (❌ junto al nombre del archivo, arriba) '
            'para volver a elegir entre los hospitales de referencia.'
        )
        hospital = None
        area = None
    else:
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

    # st.pills en vez de st.radio: mismo comportamiento (una sola opción, siempre elegida --
    # required=True se comporta como el radio, que no permite "ninguna") con el look de
    # pastillas del mockup, sin tener que pelear con el CSS interno del radio de Streamlit.
    horizonte = st.pills('Horizonte de predicción', list(logica.HORIZONTES.keys()),
                         default='3 meses', selection_mode='single', required=True)
    predecir_click = st.button('Predecir ocupación', type='primary', use_container_width=True,
                               disabled=archivo_con_error or not (hospital and area))

if predecir_click:
    st.session_state.ultimo_resultado = logica.predecir_valor(
        hospital, area, horizonte, st.session_state.df_subido
    )
    # El horizonte se guarda junto al resultado porque Streamlit re-ejecuta el script con
    # cada interacción: sin esto, mover el selector después de predecir mostraría el margen
    # de error de un horizonte distinto al que se predijo.
    st.session_state.ultimo_horizonte = horizonte
    # La sugerencia de traslado se calcula acá, una vez por predicción, y no en cada
    # re-ejecución: corre entre 30 y 90 predicciones del modelo, así que repetirla al tocar
    # cualquier widget era trabajo perdido.
    st.session_state.ultima_sugerencia = None
    resultado_nuevo = st.session_state.ultimo_resultado
    if resultado_nuevo['ok'] and resultado_nuevo['valor'] > logica.UMBRAL_TRASLADO:
        with st.spinner('Buscando alternativas de derivación…'):
            st.session_state.ultima_sugerencia = logica.sugerir_traslado(
                df_activo, resultado_nuevo['establecimiento_cod'], resultado_nuevo['area_cod'],
                resultado_nuevo['anio_obj'], resultado_nuevo['mes_obj'], maximo=2,
            )


# ── Gráfico de trayectoria (Altair, reemplaza el matplotlib de logica.graficar_trayectoria
#    solo para Streamlit -- Gradio/app.py sigue usando la versión de logica.py sin cambios) ──

# Colores del gráfico de trayectoria: el crítico (rojo) no cambia entre modos porque el
# mismo tono ya cumple contraste 3:1+ tanto en fondo claro como oscuro; el acento y la
# banda de alerta sí usan su paso más claro sobre fondo oscuro, siguiendo la paleta
# categórica de referencia (slot 1 azul, slot 4 amarillo).
_ACCENT = '#3987e5' if _ES_OSCURO else '#2a78d6'
_CRITICO = '#d03b3b'
_ALERTA = '#c98500' if _ES_OSCURO else '#eda100'
_TEXTO_CHART = '#fafafa' if _ES_OSCURO else '#0b0b0b'
_BANDA_OPACIDAD = 0.22 if _ES_OSCURO else 0.12


def _grafico_trayectoria(trayectoria):
    """Línea + área suave con bandas de zona (crítica/alerta) en vez de las líneas punteadas
    sueltas del gráfico anterior -- mismo dato que logica.graficar_trayectoria (resultado
    ['trayectoria']: lista de (anio, mes, valor, es_real)), solo una lectura más rápida de
    cuándo el área entra en zona de riesgo."""
    etiquetas = [f'{mes:02d}/{str(anio)[2:]}' for (anio, mes, _, _) in trayectoria]
    valores = [v for (_, _, v, _) in trayectoria]
    n = len(trayectoria)
    n_reales = sum(1 for (_, _, _, es_real) in trayectoria if es_real)

    df = pd.DataFrame({'Mes': etiquetas, 'Índice (%)': valores})
    df_real = df.iloc[:n_reales]
    df_proj = df.iloc[max(n_reales - 1, 0):]

    x_enc = alt.X('Mes:N', sort=etiquetas, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=10))
    y_enc = alt.Y('Índice (%):Q', scale=alt.Scale(domain=[0, 100]), title='Índice ocupacional (%)')

    bandas = alt.Chart(pd.DataFrame([
        {'y0': 90, 'y1': 100, 'Zona': 'Crítica (≥90%)'},
        {'y0': 80, 'y1': 90, 'Zona': 'Alerta (80–90%)'},
    ])).mark_rect(opacity=_BANDA_OPACIDAD).encode(
        y=alt.Y('y0:Q', scale=alt.Scale(domain=[0, 100])), y2='y1:Q',
        color=alt.Color('Zona:N', title=None,
                        scale=alt.Scale(domain=['Alerta (80–90%)', 'Crítica (≥90%)'],
                                        range=[_ALERTA, _CRITICO]),
                        legend=alt.Legend(orient='bottom', direction='horizontal')),
    )
    # Sin área de relleno bajo la curva: como los valores suelen estar altos (90%+), un wash
    # azul ahí encima tapaba justo las bandas de zona que se querían resaltar -- se probó y se
    # veía peor, así que se dejó solo la línea sobre las bandas.
    linea_real = alt.Chart(df_real).mark_line(
        point=alt.OverlayMarkDef(size=55, filled=True, color=_ACCENT), strokeWidth=2.6, color=_ACCENT
    ).encode(x=x_enc, y=y_enc)
    linea_proj = alt.Chart(df_proj).mark_line(
        point=alt.OverlayMarkDef(size=55, filled=False, color=_ACCENT), strokeWidth=2.6,
        strokeDash=[4, 4], color=_ACCENT,
    ).encode(x=x_enc, y=y_enc)
    etiqueta_final = alt.Chart(df.iloc[[-1]]).mark_text(dy=-14, fontWeight='bold', color=_TEXTO_CHART).encode(
        x=x_enc, y=y_enc, text=alt.Text('Índice (%):Q', format='.1f'))

    return (bandas + linea_real + linea_proj + etiqueta_final).properties(height=280)


# Insignia de estado: mismos 4 niveles de logica/inferencia.evaluar_semaforo, traducidos a
# una clase CSS -- solo VERDE (holgura) y ROJO (crítico) son estados que ameritan un color de
# alarma; AZUL (regular) es la operación normal y no lleva color de alerta, para que el color
# solo signifique algo cuando de verdad hay algo que atender.
_CLASE_SEMAFORO = {'🟢 VERDE': 'good', '🔵 AZUL': 'neutral', '🟡 AMARILLO': 'warning', '🔴 ROJO': 'critical'}

# Contenido ampliado de "Recomendaciones a seguir", detrás de un "Ver más" por tarjeta.
# No reemplaza ni modifica evaluar_semaforo() de inferencia.py (compartida con app.py/Gradio):
# es una capa puramente de presentación, solo para Streamlit, indexada por el mismo texto de
# 'color' que ya devuelve esa función. Se redactó con respaldo de fuentes públicas chilenas
# (Minsal, ChileAtiende, Salud Responde, SAMU) en vez de frases genéricas -- las fuentes
# quedan documentadas acá para la memoria de título, no se muestran dentro de la app:
#   - UGCC: minsal.cl/unidad-de-gestion-centralizada-de-casos-ugcc-...
#   - Ley de Urgencia: chileatiende.gob.cl/fichas/2470-...  y saludresponde.minsal.cl/ley-de-urgencia
#   - SAMU 131 / SAPU: samu.cl/preguntas-frecuentes
#   - Discharge Before Noon (práctica internacional, no exclusiva de Chile): hospitalmedicine.org
_DETALLE_RECOMENDACION = {
    '🟢 VERDE': {
        'institucional': (
            'Un índice bajo es la ventana operativa para adelantar lo que la alta demanda deja '
            'postergado: pabellones electivos complejos y mantenimientos preventivos de equipos '
            'críticos. La gestión de camas en Chile se coordina a nivel de red a través de la '
            'Unidad de Gestión Centralizada de Camas (UGCC) del Minsal, que usa justamente estos '
            'períodos de holgura para nivelar la carga entre establecimientos antes de que la '
            'demanda vuelva a subir.'
        ),
        'ciudadano': (
            'Con el hospital bajo el 60% de ocupación, la atención por consultas no urgentes '
            'debería tomar el tiempo habitual o menos. Esto no cambia lo que corresponde ante una '
            'urgencia real (riesgo de muerte o de secuela grave): la Ley de Urgencia obliga a '
            'atenderte de inmediato en la red pública de salud, sin exigirte pago antes de '
            'estabilizarte.'
        ),
    },
    '🔵 AZUL': {
        'institucional': (
            'En operación regular, la prioridad es que ninguna cama quede ocupada más tiempo del '
            'necesario. Prácticas como el "Discharge Before Noon" -- dar de alta durante la mañana '
            'a quienes ya están listos, en vez de dejarlo para la tarde -- están documentadas en la '
            'literatura internacional de gestión hospitalaria (Society of Hospital Medicine) como '
            'una forma de mejorar el flujo de camas sin necesidad de medidas de contingencia.'
        ),
        'ciudadano': (
            'El funcionamiento es el habitual, así que los tiempos de espera para lo no urgente son '
            'los normales de cada servicio. Para controles o consultas que no son urgentes, tu '
            'Cesfam o el SAPU de tu sector suelen resolver más rápido que la urgencia del hospital, '
            'que está pensada para emergencias.'
        ),
    },
    '🟡 AMARILLO': {
        'institucional': (
            'Acercarse al 90% es el momento de empezar a descomprimir antes de llegar al límite: '
            'trasladar a pacientes clínicamente estables a unidades de menor complejidad libera '
            'camas críticas para quien realmente las necesita. Es también el punto en que conviene '
            'informar el estado a la red, ya que la UGCC del Minsal monitorea la disponibilidad de '
            'camas críticas a nivel país y coordina derivaciones cuando un establecimiento empieza '
            'a saturarse.'
        ),
        'ciudadano': (
            'El hospital está cerca de su capacidad máxima, así que la atención de lo no urgente '
            'puede demorar más de lo habitual. Si no es una urgencia vital, tu Cesfam o el SAPU más '
            'cercano son alternativas más rápidas. Ante la duda de si tu situación amerita ir a '
            'urgencia, Salud Responde (600 360 7777) orienta por teléfono antes de que te traslades.'
        ),
    },
    '🔴 ROJO': {
        'institucional': (
            'Sobre el umbral crítico, la acción es habilitar camas de observación transitoria '
            '(Unidad de Corta Estadía, UCE) para absorber demanda mientras se resuelve, y activar '
            'la coordinación con la UGCC del Minsal, que evalúa y deriva pacientes hacia otros '
            'hospitales de la red pública según la disponibilidad real -- esa unidad gestionó '
            'cerca de 41.000 derivaciones a nivel nacional durante los momentos más exigentes de '
            'la pandemia.'
        ),
        'ciudadano': (
            'El hospital está, o estará, sobre su capacidad. Esto no cambia lo que la ley '
            'garantiza: si tu situación implica riesgo de muerte o de secuela grave, ningún '
            'hospital público puede negarte la atención ni exigirte pago antes de estabilizarte '
            '(Ley de Urgencia). Para todo lo que no sea una urgencia vital, conviene evaluar un '
            'centro con menor demanda dentro de la red pública, tu Cesfam o el SAPU más cercano; '
            'ante la duda, Salud Responde (600 360 7777) orienta sin que tengas que trasladarte '
            'primero.'
        ),
    },
}

with col_der, st.container(border=True):
    resultado = st.session_state.get('ultimo_resultado')
    if not resultado:
        st.info('Elige un hospital, un área y un horizonte, y presiona "Predecir ocupación".')
    elif not resultado['ok']:
        (st.error if resultado['es_error'] else st.info)(resultado['mensaje'])
    else:
        semaforo = resultado['semaforo']
        tipo = 'proyección' if resultado['es_prediccion'] else 'dato real'
        etiqueta_mes = f"{resultado['mes_obj']:02d}/{resultado['anio_obj']}"
        clase = _CLASE_SEMAFORO.get(semaforo['color'], 'neutral')

        st.markdown(
            f'<span class="badge badge-{clase}"><span class="dot"></span>{semaforo["nivel"]}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="hero-num">{resultado["valor"]:.1f}<small>%</small></div>'
            f'<div class="hero-label">Índice ocupacional · {tipo} · {etiqueta_mes}</div>',
            unsafe_allow_html=True,
        )

        # Punto 1 de las anotaciones: mostrar R²/MAE tal cual es jerga que no le dice nada a
        # alguien sin formación en estadística. En vez de eso, se traduce el error del modelo
        # a un margen en las mismas unidades que ve el usuario (puntos porcentuales de
        # ocupación), como una letra chica bajo el resultado. El expander de abajo explica
        # ese margen en una sola idea, sin R² ni palabras como "validación" o "entrenamiento"
        # -- quien quiera el detalle estadístico completo puede pedirlo directamente.
        # Punto del profesor guía: el porcentaje por sí solo no dice cuántas camas son.
        # logica.predecir_valor() ya deja el cálculo hecho en resultado['camas'] (None si no
        # se conoce la dotación del área). Se habla de promedio del mes porque el índice del
        # REM son días-cama ocupados sobre días-cama disponibles durante el mes, no una foto.
        camas = resultado.get('camas')
        if camas:
            st.markdown(
                f'<span class="camas-linea">🛏️ De {camas["disponibles"]:.0f} camas del área, '
                f'unas {camas["ocupadas"]:.0f} ocupadas y {camas["libres"]:.0f} libres '
                f'(promedio del mes)</span>',
                unsafe_allow_html=True,
            )

        horizonte_predicho = st.session_state.get('ultimo_horizonte', horizonte)
        if logica.MAE is not None and resultado['es_prediccion']:
            mae_horizonte = logica.MAE_POR_HORIZONTE.get(horizonte_predicho, logica.MAE)
            es_global = horizonte_predicho not in logica.MAE_POR_HORIZONTE
            st.markdown(f'<span class="margin-pill">📏 Margen de error típico: '
                        f'±{mae_horizonte:.1f} puntos porcentuales</span>', unsafe_allow_html=True)
            with st.expander('¿Qué tan confiable es esta predicción?'):
                nota_global = (f'\n\n*(Este margen todavía es general para el modelo, no específico '
                                f'para {horizonte_predicho}.)*' if es_global else '')
                st.markdown(
                    f"El modelo no acierta siempre el número exacto: en pruebas anteriores, se equivocó "
                    f"en promedio por **±{mae_horizonte:.1f} puntos**. Así que el **{resultado['valor']:.1f}%** "
                    f"de arriba es una buena estimación, pero el valor real podría ser un poco más alto o "
                    f"más bajo.{nota_global}"
                )

        st.altair_chart(_grafico_trayectoria(resultado['trayectoria']), use_container_width=True,
                        theme='streamlit')
        st.caption('— Real (línea sólida) ┄ Proyectado (línea punteada)')

        # ── Explicación en palabras simples, debajo del gráfico ──────────────────
        # Todo el contenido sale de explicacion.py: los números los calcula Python y la IA
        # solo los redacta, con una verificación que descarta el texto si menciona una cifra
        # que no se le entregó. Si no hay clave de API, si falla la llamada o si el texto no
        # pasa esa verificación, se muestra la versión determinista -- nunca queda vacío.
        hechos = explicacion.construir_hechos(
            resultado,
            sugerencia=st.session_state.get('ultima_sugerencia'),
            mae=logica.MAE_POR_HORIZONTE.get(horizonte_predicho, logica.MAE),
        )
        explicada = explicacion.explicar(hechos)
        st.markdown(
            f'<div class="caja-explica"><div class="titulo">Qué significa esto</div>'
            f'<p>{html.escape(explicada["texto"])}</p></div>',
            unsafe_allow_html=True,
        )
        if explicada['origen'] == 'plantilla' and explicacion.hay_ia_disponible():
            # solo interesa avisar cuando la IA estaba configurada y aun así no se usó: ahí
            # hay algo que revisar (clave, red, o un texto descartado por la verificación)
            with st.expander('Por qué este texto no se generó con IA'):
                st.caption(explicada['detalle'])


# ── Recomendaciones: sección de ancho completo, debajo del botón y del gráfico ─────

def _titulo_centrado(texto, nivel=2):
    st.markdown(f"<h{nivel} style='text-align:center'>{html.escape(texto)}</h{nivel}>",
                unsafe_allow_html=True)


# Colores del gráfico de alternativas (forma de 'énfasis'): el área recomendada en azul,
# la más desocupada en naranja y el resto en gris. Ahora sí sigue a _ES_OSCURO (antes se
# fijaba siempre en paleta clara); el texto pasa a tinta clara sobre fondo oscuro para que
# las etiquetas de valor y la línea de umbral sigan siendo legibles.
_COLORES_ALTERNATIVAS = (
    {'Recomendada': '#3987e5', 'Más baja': '#d95926', 'Otras': '#8b8a84', 'texto': '#f2f1ee'}
    if _ES_OSCURO else
    {'Recomendada': '#2a78d6', 'Más baja': '#eb6834', 'Otras': '#a3a29c', 'texto': '#31333f'}
)
# pista/track del "medidor": un paso gris apenas por encima de la superficie, para que se
# note como fondo sin competir con la barra de color de encima
_PISTA_COLOR = '#2b303a' if _ES_OSCURO else '#eceae4'


def _etiqueta_area(nombre):
    """'Área de Hospitalización de ...' -> 'Hospitalización de ...': el prefijo se repite en
    todas y ocupa el ancho que necesita el nombre. El nombre completo va en el tooltip."""
    corto = re.sub(r'^Área (de )?', '', nombre).strip()
    return corto[:1].upper() + corto[1:]


def _grafico_alternativas(areas):
    """'Medidor' horizontal (pista gris de 0-100 + barra de color encima) de las áreas bajo
    el umbral de un hospital sugerido, de menor a mayor -- reemplaza la barra plana anterior,
    que varios proyectos de la galería de Streamlit resuelven así: la pista de fondo deja ver
    de un vistazo cuánto le queda a cada área para llegar a 100%, no solo cuánto lleva. Se
    destacan la recomendada (areas[0]: logica.sugerir_traslado la entrega primero, la misma
    área consultada o, si no la tiene, la más desocupada) y la más baja. Al pasar el mouse por
    una fila, esa barra se resalta (borde + opacidad) para que la comparación entre áreas se
    sienta interactiva y no una imagen estática."""
    colores = _COLORES_ALTERNATIVAS
    recomendada = areas[0]['area']
    minimo = min(a['valor'] for a in areas)

    filas = []
    for a in areas:
        # si la recomendada es además la más baja, gana 'Recomendada'
        grupo = ('Recomendada' if a['area'] == recomendada
                 else 'Más baja' if a['valor'] == minimo else 'Otras')
        filas.append({'Área': a['area'], 'Etiqueta': _etiqueta_area(a['area']),
                      'Índice (%)': round(a['valor'], 1), 'Grupo': grupo,
                      'Texto': f"{a['valor']:.1f}%", 'Pista': 100})
    datos = pd.DataFrame(filas)
    orden = datos.sort_values('Índice (%)')['Etiqueta'].tolist()
    grupos = [g for g in ('Recomendada', 'Más baja', 'Otras') if g in set(datos['Grupo'])]

    # selección de hover: fields=['Etiqueta'] la comparten las 3 capas de abajo, así que basta
    # pasar el mouse por cualquier punto de la fila (incluida la pista vacía) para resaltarla
    hover = alt.selection_point(fields=['Etiqueta'], on='mouseover', nearest=True, empty=False)

    y_enc = alt.Y('Etiqueta:N', sort=orden, title=None,
                  axis=alt.Axis(labelLimit=230, labelOverlap=False, ticks=False, domain=False))
    base = alt.Chart(datos).encode(y=y_enc)

    # pista: el 100% completo en gris muy claro, así cada barra de color se lee como "cuánto
    # de la capacidad ya está tomada" en vez de una barra suelta sin referencia de escala
    pista = base.mark_bar(cornerRadiusEnd=4, height={'band': 0.62}, color=_PISTA_COLOR).encode(
        x=alt.X('Pista:Q', scale=alt.Scale(domain=[0, 100]), title=None))

    barras = base.mark_bar(cornerRadiusEnd=4, height={'band': 0.62}).encode(
        x=alt.X('Índice (%):Q', scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(values=[0, 20, 40, 60, 80, 100],
                              title=f'Índice ocupacional proyectado (%) · punteado: umbral {logica.UMBRAL_TRASLADO:.0f}%')),
        color=alt.Color('Grupo:N', title=None,
                        scale=alt.Scale(domain=grupos, range=[colores[g] for g in grupos]),
                        legend=alt.Legend(orient='bottom', direction='horizontal')),
        opacity=alt.condition(hover, alt.value(1.0), alt.value(0.82)),
        stroke=alt.condition(hover, alt.value(colores['texto']), alt.value(None)),
        strokeWidth=alt.condition(hover, alt.value(1.4), alt.value(0)),
    )
    # capa transparente del ancho completo de la pista: hace que el hover reaccione en toda la
    # fila (no solo sobre el tramo corto de la barra de color) y trae el tooltip con esa área
    captura = base.mark_bar(height={'band': 0.62}, opacity=0.001).encode(
        x=alt.X('Pista:Q', scale=alt.Scale(domain=[0, 100])),
        tooltip=[alt.Tooltip('Área:N'), alt.Tooltip('Índice (%):Q', format='.1f'),
                 alt.Tooltip('Grupo:N', title='Destacada como')],
    ).add_params(hover)
    # todas las áreas listadas están bajo 85%, así que el valor cabe a la derecha de la barra
    valores = base.mark_text(align='left', dx=4, fontWeight='bold', color=colores['texto']).encode(
        x='Índice (%):Q', text='Texto:N')
    linea = alt.Chart(pd.DataFrame({'x': [logica.UMBRAL_TRASLADO]})).mark_rule(
        strokeDash=[4, 4], opacity=0.7, color=colores['texto']).encode(x='x:Q')
    # alto por fila (step) y no total: así cada área tiene su espacio aunque sean muchas,
    # y los ejes y la leyenda se suman aparte en vez de comerse el alto de las barras
    return (pista + barras + captura + valores + linea).properties(height=alt.Step(40))


resultado = st.session_state.get('ultimo_resultado')
if resultado and resultado['ok']:
    semaforo = resultado['semaforo']
    etiqueta_mes = f"{resultado['mes_obj']:02d}/{resultado['anio_obj']}"

    st.write('')
    with st.container(border=True):
        _titulo_centrado('Recomendaciones a seguir')
        detalle = _DETALLE_RECOMENDACION.get(semaforo['color'])
        col_reco1, col_reco2 = st.columns(2)
        with col_reco1:
            st.markdown(
                f'<div class="reco-card role-inst"><div class="reco-icon role-inst">🏛️</div>'
                f'<div><b>Gestión hospitalaria</b><p>{html.escape(semaforo["institucional"])}</p></div></div>',
                unsafe_allow_html=True,
            )
            if detalle:
                with st.expander('Ver más'):
                    st.markdown(detalle['institucional'])
        with col_reco2:
            st.markdown(
                f'<div class="reco-card role-cit"><div class="reco-icon role-cit">🧑‍⚕️</div>'
                f'<div><b>Para la ciudadanía</b><p>{html.escape(semaforo["ciudadano"])}</p></div></div>',
                unsafe_allow_html=True,
            )
            if detalle:
                with st.expander('Ver más'):
                    st.markdown(detalle['ciudadano'])

    # Motor de derivación: si el área queda sobre el umbral (85%), se comparan los 2
    # hospitales del mismo servicio de salud que para ese mismo mes proyectan menos.
    # Toda la lógica está en logica.sugerir_traslado(); acá solo se presenta.
    if resultado['valor'] > logica.UMBRAL_TRASLADO:
        st.write('')
        with st.container(border=True):
            _titulo_centrado('Sugerencia de hospitales para traslado')
            # ya calculada al momento de predecir (ver más arriba); si por algún motivo no
            # está en la sesión, se calcula acá para no dejar la sección vacía
            sugerencia = st.session_state.get('ultima_sugerencia')
            if sugerencia is None:
                with st.spinner('Buscando alternativas de derivación…'):
                    sugerencia = logica.sugerir_traslado(
                        df_activo, resultado['establecimiento_cod'], resultado['area_cod'],
                        resultado['anio_obj'], resultado['mes_obj'], maximo=2,
                    )
                st.session_state.ultima_sugerencia = sugerencia
            ambito = f"Servicio de salud {html.escape(str(sugerencia['servicio']))}"
            if sugerencia.get('region'):
                ambito += f" · región {html.escape(str(sugerencia['region']))}"
            st.markdown(f"<p style='text-align:center'>{ambito} "
                        f"· proyección {etiqueta_mes} · áreas bajo {logica.UMBRAL_TRASLADO:.0f}% "
                        f"· ordenadas por cercanía</p>",
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
                    # dónde queda y a qué distancia: el dato que convierte una lista de
                    # nombres en una alternativa evaluable
                    ubicacion = []
                    if hospital_alt.get('comuna'):
                        ubicacion.append(f"comuna de {hospital_alt['comuna']}")
                    if hospital_alt.get('distancia_texto'):
                        ubicacion.append(f"a {hospital_alt['distancia_texto']} en línea recta")
                    if ubicacion:
                        st.markdown(f"<p style='text-align:center'>📍 {html.escape(' · '.join(ubicacion))}</p>",
                                    unsafe_allow_html=True)
                    st.markdown(f"**Área recomendada:** {recomendada['area']} — {recomendada['valor']:.1f}%")
                    camas_alt = hospital_alt.get('camas')
                    if camas_alt:
                        st.markdown(f"🛏️ De {camas_alt['disponibles']:.0f} camas, unas "
                                    f"{camas_alt['libres']:.0f} estarían libres (promedio del mes)")
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
