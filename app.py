"""
App de Gradio: predicción de ocupación hospitalaria para cualquier hospital público
de la Red REM 20 (o uno nuevo que suba su propio historial). Pensada para desplegarse
como un único Hugging Face Space -- un solo proceso, sin API separada.

Toda la lógica de negocio (carga del modelo, combinar lo subido con lo conocido,
armar la predicción) vive en logica.py, compartida con streamlit_app.py -- acá solo
se traduce esa lógica a componentes de Gradio.

Cómo se prueba localmente: `python app.py` (crea la URL local que imprime Gradio).
"""
import gradio as gr

import logica

HOSPITALES_REFERENCIA = logica.HOSPITALES_REFERENCIA
AREAS_REFERENCIA = logica.AREAS_REFERENCIA
HORIZONTES = logica.HORIZONTES


# ── Callbacks de subida de archivo ────────────────────────────────────────────────

def manejar_subida(archivo):
    if archivo is None:
        return (
            None,
            gr.update(choices=HOSPITALES_REFERENCIA, value=None),
            gr.update(choices=AREAS_REFERENCIA, value=None),
            '',
        )
    df_subido, mensaje, es_error = logica.cargar_archivo_subido(archivo.name)
    clase = 'msg-error' if es_error else 'msg-ok'
    html = f'<div class="msg {clase}">{mensaje}</div>'
    if es_error:
        return None, gr.update(choices=HOSPITALES_REFERENCIA, value=None), gr.update(choices=AREAS_REFERENCIA, value=None), html

    hospitales_subidos = logica.hospitales_disponibles(df_subido)
    return (
        df_subido,
        gr.update(choices=hospitales_subidos, value=hospitales_subidos[0] if hospitales_subidos else None),
        gr.update(choices=AREAS_REFERENCIA, value=None),
        html,
    )


def buscar_hospital(texto, df_subido):
    coincidencias = logica.buscar_hospital(texto, df_subido)
    if not coincidencias:
        return gr.update(choices=[], value=None)
    return gr.update(choices=coincidencias, value=coincidencias[0])


def quitar_archivo():
    return (
        None,
        gr.update(choices=HOSPITALES_REFERENCIA, value=None),
        gr.update(choices=AREAS_REFERENCIA, value=None),
        '',
    )


# ── Callback de predicción ────────────────────────────────────────────────────────

def _render_semaforo(resultado):
    semaforo = resultado['semaforo']
    tipo = 'proyección' if resultado['es_prediccion'] else 'dato real'
    etiqueta_mes = f"{resultado['mes_obj']:02d}/{resultado['anio_obj']}"
    return f'''
    <div class="semaforo">
      <div class="semaforo-badge">{semaforo["color"]} — {semaforo["nivel"]}</div>
      <div class="semaforo-valor">{resultado["valor"]:.1f}%</div>
      <div class="semaforo-nota">{tipo} · {etiqueta_mes}</div>
    </div>
    '''


def _render_recos(semaforo):
    return f'''
    <div class="recos">
      <div class="reco"><span class="tag">Para gestión hospitalaria</span><p>{semaforo["institucional"]}</p></div>
      <div class="reco"><span class="tag">Para la ciudadanía</span><p>{semaforo["ciudadano"]}</p></div>
    </div>
    '''


def predecir(hospital_nombre, area_nombre, horizonte_label, df_subido):
    resultado = logica.predecir_valor(hospital_nombre, area_nombre, horizonte_label, df_subido)
    if not resultado['ok']:
        clase = 'msg-error' if resultado['es_error'] else 'msg-info'
        return f'<div class="msg {clase}">{resultado["mensaje"]}</div>', None, ''

    return (
        _render_semaforo(resultado),
        logica.graficar_trayectoria(resultado['trayectoria']),
        _render_recos(resultado['semaforo']),
    )


# ── Interfaz ───────────────────────────────────────────────────────────────────────

CSS = '''
.msg { padding: .6rem .8rem; border-radius: 8px; font-size: .9rem; margin-top: .5rem; }
.msg-ok { background: #e3f1ee; color: #0e6e64; }
.msg-error { background: #fbe9e7; color: #a93226; }
.msg-info { background: #eef2f0; color: #55655f; }
.semaforo { padding: .9rem 1rem; border-radius: 10px; background: #eef2f0; }
.semaforo-badge { font-weight: 700; font-size: .95rem; }
.semaforo-valor { font-size: 2rem; font-weight: 700; font-family: monospace; margin-top: .3rem; }
.semaforo-nota { font-size: .8rem; color: #55655f; }
.recos { display: flex; gap: .8rem; margin-top: .8rem; flex-wrap: wrap; }
.reco { flex: 1; min-width: 220px; background: #fff; border: 1px solid #d7ddda; border-radius: 8px; padding: .7rem .8rem; }
.reco .tag { font-size: .68rem; font-weight: 700; text-transform: uppercase; color: #0e6e64; display: block; margin-bottom: .3rem; }
.reco p { margin: 0; font-size: .85rem; }
'''

with gr.Blocks(title='Predicción de Ocupación Hospitalaria') as demo:
    estado_archivo = gr.State(None)  # dataframe subido por ESTA sesión -- nunca global

    gr.Markdown('## Predicción de Ocupación Hospitalaria\nModelo estandarizado — cualquier hospital público de la Red REM 20')

    with gr.Row():
        with gr.Column():
            gr.Markdown('**1. Cargar datos (opcional)**')
            archivo_input = gr.File(label='CSV de un hospital (mismo formato REM 20)', file_types=['.csv'])
            estado_html = gr.HTML()

            gr.Markdown('**2. Elegir qué predecir**')
            buscar_tb = gr.Textbox(label='Buscar hospital', placeholder='ej. "hospital sotero del rio" (sin tildes, nombre incompleto)')
            hospital_dd = gr.Dropdown(choices=HOSPITALES_REFERENCIA, label='Hospital', value=None)
            area_dd = gr.Dropdown(choices=AREAS_REFERENCIA, label='Área funcional', value=None)
            horizonte_radio = gr.Radio(list(HORIZONTES.keys()), value='3 meses', label='Horizonte de predicción')
            predecir_btn = gr.Button('Predecir ocupación', variant='primary')

        with gr.Column():
            semaforo_html = gr.HTML()
            trayectoria_plot = gr.Plot(label='Trayectoria')
            recos_html = gr.HTML()

    archivo_input.upload(
        fn=manejar_subida,
        inputs=[archivo_input],
        outputs=[estado_archivo, hospital_dd, area_dd, estado_html],
    )
    archivo_input.clear(
        fn=quitar_archivo,
        outputs=[estado_archivo, hospital_dd, area_dd, estado_html],
    )
    buscar_tb.submit(
        fn=buscar_hospital,
        inputs=[buscar_tb, estado_archivo],
        outputs=[hospital_dd],
    )
    predecir_btn.click(
        fn=predecir,
        inputs=[hospital_dd, area_dd, horizonte_radio, estado_archivo],
        outputs=[semaforo_html, trayectoria_plot, recos_html],
    )

if __name__ == '__main__':
    demo.launch(css=CSS)
