# -*- coding: utf-8 -*-
"""
Explicación en lenguaje sencillo de lo que muestra el gráfico de predicción.

Responde la sugerencia del profesor guía: un cuadro bajo el gráfico que traduzca el
porcentaje a algo concreto -- camas físicas -- y lo cuente en palabras simples.

La arquitectura tiene una regla que ordena todo lo demás:

    LOS NÚMEROS LOS CALCULA PYTHON. LA IA SOLO LOS REDACTA.

construir_hechos() arma un diccionario con cifras ya calculadas (índice, camas, mes,
margen de error, alternativas). A partir de ahí hay dos caminos para convertir esos hechos
en un párrafo:

  1. texto_plantilla(hechos)  -- determinista, sin red, sin clave, sin costo. Siempre
     funciona y siempre dice lo mismo para los mismos datos.
  2. texto_ia(hechos)         -- le pide a Claude que redacte esos mismos hechos con mejor
     prosa. Solo se usa si hay clave configurada, si la API responde, y si el texto pasa
     la verificación de números de abajo.

explicar() intenta (2) y cae a (1) ante cualquier problema. O sea: la app nunca queda sin
explicación. Sin clave de API, sin internet, con la API caída, con la cuenta sin saldo o
dentro de un año cuando nadie se acuerde de renovar nada, la sección sigue mostrando el
párrafo determinista. Eso es lo que permite presentar el proyecto en cualquier momento sin
depender de un servicio externo.

VERIFICACIÓN DE NÚMEROS (por qué se puede confiar en el texto de la IA)
Un modelo de lenguaje puede inventar una cifra, y en un contexto de salud eso no es
aceptable. Por eso _verificar_numeros() extrae TODOS los números del texto generado y los
compara contra la lista de valores que se le entregaron. Si aparece aunque sea uno que no
calza, el texto completo se descarta y se usa la plantilla. No se corrige ni se reintenta:
se descarta. El usuario nunca ve una cifra que no haya salido de los datos.

CONFIGURAR LA CLAVE (opcional)
En Streamlit Community Cloud: Manage app -> Settings -> Secrets, y pegar:
    ANTHROPIC_API_KEY = "sk-ant-..."
En local: exportar la variable de entorno ANTHROPIC_API_KEY.
Sin clave, el módulo no intenta llamar a la API y usa la plantilla directamente.
La clave nunca se escribe en el código ni se sube al repositorio.
"""
import math
import os
import re

# Modelo que redacta. Claude Opus 5.5 es el que da mejor prosa en español; si el costo
# llega a importar, 'claude-sonnet-5-5' o 'claude-haiku-4-5' sirven igual para esta tarea
# (es redacción corta a partir de cifras ya calculadas) y son más baratos. Cambiar solo
# esta constante: todo lo demás, incluida la verificación de números, queda igual.
MODELO_IA = 'claude-opus-5-5'

# El párrafo pedido son 3 o 4 frases. El tope alto es para que el modelo tenga aire y no
# devuelva una respuesta cortada a la mitad (una respuesta truncada se descarta igual).
MAX_TOKENS = 2000

# Si la API no contestó en este tiempo, se muestra la plantilla. Vale más una explicación
# instantánea y correcta que una mejor que deje la página pensando.
TIMEOUT_SEGUNDOS = 20

# Caché en memoria del proceso: los mismos hechos dan el mismo texto, así que no se paga ni
# se espera dos veces por la misma consulta. Se vacía al reiniciar la app, que es justo lo
# que se quiere (si cambian los datos, se vuelve a redactar).
_CACHE = {}
_CACHE_MAXIMO = 256

_CLAVE_EXPLICITA = None

MESES = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
         'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']


def configurar_clave(clave):
    """Entrega la clave de API desde fuera (en Streamlit, st.secrets). Si no se llama, se
    usa la variable de entorno ANTHROPIC_API_KEY."""
    global _CLAVE_EXPLICITA
    _CLAVE_EXPLICITA = clave or None


def _clave():
    return _CLAVE_EXPLICITA or os.environ.get('ANTHROPIC_API_KEY') or None


def hay_ia_disponible():
    """True si hay una clave configurada y la librería instalada. Sirve para que la
    interfaz no ofrezca un botón que no va a funcionar."""
    if not _clave():
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


# ── Los hechos: todo lo que se puede decir, ya calculado ──────────────────────────

def _nombre_mes(anio, mes):
    return f'{MESES[mes - 1]} de {anio}'


def _area_legible(nombre):
    """'Área Cuidados Intermedios Adultos' -> 'cuidados intermedios adultos'.

    Dos cosas: se saca el prefijo 'Área', porque las frases ya dicen 'el área de ...' y
    quedaba 'el área Área Cuidados Intermedios Adultos'; y se pasa todo a minúscula, porque
    el REM escribe los nombres en mayúscula inicial de cada palabra y dentro de una frase
    eso se lee como un título pegado. Ninguno de los 29 nombres del REM contiene nombres
    propios, así que bajarlos entero es seguro. La interfaz sigue mostrando el nombre
    oficial completo: esto es solo para el párrafo explicativo."""
    texto = re.sub(r'^Área\s+(de\s+)?', '', str(nombre or '')).strip()
    return texto.lower() if texto else str(nombre or '')


def _coma(valor, decimales=1):
    """Número en formato chileno (coma decimal). 93.4 -> '93,4'."""
    return f'{valor:.{decimales}f}'.replace('.', ',')


def construir_hechos(resultado, sugerencia=None, mae=None):
    """Reúne, ya calculado, todo lo que la explicación puede mencionar.

    `resultado` es lo que devuelve logica.predecir_valor(); `sugerencia`, lo que devuelve
    logica.sugerir_traslado() cuando el área quedó sobre el umbral (o None); `mae`, el
    margen de error del modelo para ese horizonte, en puntos porcentuales.

    Devuelve un diccionario plano. Nada de lo que no esté acá puede aparecer en el texto."""
    camas = resultado.get('camas')
    trayectoria = resultado.get('trayectoria') or []
    reales = [v for (_, _, v, es_real) in trayectoria if es_real]

    hechos = {
        'hospital': resultado.get('hospital_nombre'),
        'comuna': resultado.get('comuna'),
        'area': resultado.get('area_nombre'),
        'area_legible': _area_legible(resultado.get('area_nombre')),
        'mes_texto': _nombre_mes(resultado['anio_obj'], resultado['mes_obj']),
        'mes_numero': resultado['mes_obj'],
        'anio': resultado['anio_obj'],
        'horizonte': resultado.get('horizonte_label'),
        'indice': round(resultado['valor'], 1),
        'es_prediccion': resultado.get('es_prediccion', True),
        'nivel': resultado['semaforo']['nivel'],
        'color': resultado['semaforo']['color'],
        'umbral': 85.0,
        'mae': round(mae, 1) if mae is not None else None,
        'camas_disponibles': round(camas['disponibles']) if camas else None,
        'camas_ocupadas': round(camas['ocupadas']) if camas else None,
        'camas_libres': round(camas['libres']) if camas else None,
        'ultimo_real': round(reales[-1], 1) if reales else None,
        'meses_reales': len(reales),
        'alternativas': [],
    }

    # tendencia: solo sube / baja / se mantiene, sin inventar una pendiente
    if len(reales) >= 3:
        cambio = reales[-1] - reales[0]
        hechos['tendencia'] = 'subiendo' if cambio > 2 else 'bajando' if cambio < -2 else 'estable'
    else:
        hechos['tendencia'] = None

    if sugerencia and not sugerencia.get('sin_alternativas'):
        for hospital in sugerencia['hospitales']:
            area_rec = hospital['areas'][0]
            camas_alt = hospital.get('camas')
            hechos['alternativas'].append({
                'nombre': hospital['nombre'],
                'comuna': hospital.get('comuna'),
                'distancia_texto': hospital.get('distancia_texto') or '',
                'distancia_km': (round(hospital['distancia_km'], 1)
                                 if hospital.get('distancia_km') is not None else None),
                'area': area_rec['area'],
                'area_legible': _area_legible(area_rec['area']),
                'indice': round(area_rec['valor'], 1),
                'misma_area': hospital.get('misma_area', False),
                'camas_libres': round(camas_alt['libres']) if camas_alt else None,
            })
    return hechos


# ── Camino 1: la plantilla determinista ───────────────────────────────────────────

def texto_plantilla(hechos):
    """Explicación armada en Python, sin IA. Es la que se muestra cuando no hay clave de
    API o cuando la llamada falla, y la que se usa para comparar contra el texto generado.

    Se escribe en frases cortas y sin jerga a propósito: nada de R², 'validación' ni
    'intervalo de confianza'."""
    partes = []
    cuando = 'proyecta' if hechos['es_prediccion'] else 'registró'
    partes.append(
        f"Para {hechos['mes_texto']}, el modelo {cuando} que el área de "
        f"{hechos['area_legible']} en el {hechos['hospital']} va a estar ocupada en un "
        f"{_coma(hechos['indice'])}%."
    )

    if hechos['camas_disponibles']:
        libres = hechos['camas_libres']
        if libres >= 1:
            frase_libres = f"quedarían alrededor de {libres} libres"
        else:
            frase_libres = "prácticamente no quedarían camas libres"
        partes.append(
            f"En camas, eso significa que de las {hechos['camas_disponibles']} que tiene "
            f"el área, unas {hechos['camas_ocupadas']} estarían ocupadas y {frase_libres}, "
            f"en promedio durante el mes."
        )

    if hechos['indice'] >= hechos['umbral']:
        partes.append(
            f"Es un nivel alto: sobre {_coma(hechos['umbral'], 0)}% se considera que el "
            f"servicio está al límite y conviene tomar medidas antes de que se sature."
        )
    elif hechos['indice'] >= 60:
        partes.append("Es un nivel de funcionamiento normal, sin holgura de sobra pero sin apremio.")
    else:
        partes.append("Es un nivel holgado: hay capacidad disponible con margen.")

    if hechos['tendencia'] == 'subiendo':
        partes.append("Además viene subiendo en los últimos meses.")
    elif hechos['tendencia'] == 'bajando':
        partes.append("La buena noticia es que viene bajando en los últimos meses.")

    if hechos['alternativas']:
        alt = hechos['alternativas'][0]
        cercania = f", a {alt['distancia_texto']} de distancia," if alt['distancia_texto'] else ''
        partes.append(
            f"El centro más cercano con capacidad es el {alt['nombre']}{cercania} "
            f"donde el área de {alt['area_legible']} estaría en {_coma(alt['indice'])}%."
        )

    if hechos['mae'] is not None:
        partes.append(
            f"El modelo se equivoca en promedio unos {_coma(hechos['mae'])} puntos, "
            f"así que el valor real puede ser algo más alto o más bajo."
        )

    return ' '.join(partes)


# ── Camino 2: la IA redacta los mismos hechos ─────────────────────────────────────

SISTEMA = (
    "Eres parte de una herramienta chilena que proyecta la ocupación de camas en "
    "hospitales públicos. Tu única tarea es redactar, en español de Chile, una explicación "
    "breve y clara de unas cifras que ya vienen calculadas.\n\n"
    "Reglas estrictas:\n"
    "1. Usa SOLO los números que te entrego. No calcules, derives ni estimes ninguna cifra "
    "nueva: ni porcentajes adicionales, ni diferencias, ni proyecciones propias.\n"
    "2. No inventes causas (brotes, campañas de invierno, decisiones del hospital). No "
    "sabes por qué pasa lo que pasa, solo qué dicen los datos.\n"
    "3. No des indicaciones clínicas ni le digas a nadie qué hacer con un paciente.\n"
    "4. Lenguaje cotidiano: nada de 'índice ocupacional', 'modelo predictivo', 'variable', "
    "'margen de error estadístico' ni términos técnicos. Escribe como le explicarías la "
    "situación a un familiar que no trabaja en salud.\n"
    "5. De 3 a 4 frases, un solo párrafo. Sin títulos, sin viñetas, sin negritas.\n"
    "6. Habla de camas, no solo de porcentajes: es lo que hace entendible la cifra.\n"
    "7. Si el nivel es alto, dilo sin alarmismo ni dramatismo."
)


def _prompt(hechos):
    """Los hechos, en texto, para que el modelo los redacte. Es deliberadamente una lista
    de datos y no una instrucción abierta: mientras menos espacio haya para improvisar,
    menos posibilidad de que aparezca un número que no corresponde."""
    lineas = [
        f"Hospital: {hechos['hospital']}" + (f" (comuna de {hechos['comuna']})" if hechos['comuna'] else ''),
        f"Área: {hechos['area_legible']}",
        f"Mes proyectado: {hechos['mes_texto']}",
        f"Ocupación proyectada: {_coma(hechos['indice'])}%",
        f"Nivel: {hechos['nivel']}",
        f"Umbral de saturación usado en Chile: {_coma(hechos['umbral'], 0)}%",
    ]
    if hechos['camas_disponibles']:
        lineas += [
            f"Camas del área: {hechos['camas_disponibles']}",
            f"Camas que estarían ocupadas: {hechos['camas_ocupadas']}",
            f"Camas que quedarían libres: {hechos['camas_libres']}",
            "(las camas libres son un promedio del mes, no un conteo de un instante)",
        ]
    if hechos['ultimo_real'] is not None:
        lineas.append(f"Último valor realmente medido: {_coma(hechos['ultimo_real'])}%")
    if hechos['tendencia']:
        lineas.append(f"Tendencia de los últimos meses: {hechos['tendencia']}")
    if hechos['mae'] is not None:
        lineas.append(f"El modelo se equivoca en promedio: {_coma(hechos['mae'])} puntos")
    for alt in hechos['alternativas']:
        detalle = f"- Alternativa cercana: {alt['nombre']}"
        if alt['comuna']:
            detalle += f" (comuna de {alt['comuna']})"
        if alt['distancia_texto']:
            detalle += f", a {alt['distancia_texto']}"
        detalle += f", su área de {alt['area_legible']} estaría en {_coma(alt['indice'])}%"
        if alt['camas_libres'] is not None:
            detalle += f" con unas {alt['camas_libres']} camas libres"
        lineas.append(detalle)

    return ('Redacta la explicación con estos datos:\n\n' + '\n'.join(lineas)
            + '\n\nRecuerda: solo estos números, de 3 a 4 frases, un párrafo, lenguaje cotidiano.')


def _valores_permitidos(hechos):
    """Todos los números que el texto tiene derecho a mencionar.

    Se agregan las variantes redondeadas porque es legítimo que el texto diga 'unas 34
    camas' cuando el dato es 33,6; lo que no es legítimo es que aparezca un 41 que no sale
    de ninguna parte."""
    valores = set()

    def agregar(x):
        if x is None:
            return
        x = float(x)
        for v in (x, round(x), math.floor(x), math.ceil(x), round(x, 1)):
            valores.add(float(v))

    for clave in ('indice', 'umbral', 'mae', 'camas_disponibles', 'camas_ocupadas',
                  'camas_libres', 'ultimo_real', 'mes_numero', 'anio', 'meses_reales'):
        agregar(hechos.get(clave))

    for alt in hechos['alternativas']:
        agregar(alt['indice'])
        agregar(alt['camas_libres'])
        agregar(alt['distancia_km'])
        # la distancia se muestra formateada ('232 m', '2,7 km'): se permite el número
        # tal como aparece en ese texto
        for numero in re.findall(r'\d+(?:[.,]\d+)?', alt['distancia_texto'] or ''):
            agregar(numero.replace('.', '').replace(',', '.'))

    # el horizonte ('2 meses') y los números de las etiquetas de nivel
    for numero in re.findall(r'\d+', str(hechos.get('horizonte') or '')):
        agregar(numero)
    return valores


def _verificar_numeros(texto, hechos):
    """(ok, numero_intruso). Falla si el texto menciona un número que no se le entregó.

    Tolerancia de 0,6 para absorber el redondeo ('unas 34 camas' cuando el dato es 33,6).
    Los años entre 2000 y 2100 se aceptan siempre, porque una fecha escrita en palabras es
    inofensiva y ya viene del mes objetivo."""
    permitidos = _valores_permitidos(hechos)
    for crudo in re.findall(r'\d+(?:[.,]\d+)?', texto):
        try:
            valor = float(crudo.replace('.', '').replace(',', '.')) if ',' in crudo else float(crudo)
        except ValueError:
            continue
        if 2000 <= valor <= 2100 and valor == int(valor):
            continue
        if not any(abs(valor - p) <= 0.6 for p in permitidos):
            return False, crudo
    return True, None


def texto_ia(hechos):
    """(texto, detalle). texto es None si no se pudo generar o si no pasó la verificación;
    detalle explica por qué, para poder mostrarlo en un expander de diagnóstico.

    Ningún fallo de acá debe llegar nunca al usuario como un error: todos terminan en un
    texto None y la interfaz muestra la plantilla."""
    clave = _clave()
    if not clave:
        return None, 'No hay clave de API configurada (ANTHROPIC_API_KEY).'

    try:
        import anthropic
    except ImportError:
        return None, 'La librería anthropic no está instalada.'

    try:
        cliente = anthropic.Anthropic(api_key=clave, timeout=TIMEOUT_SEGUNDOS, max_retries=1)
        respuesta = cliente.messages.create(
            model=MODELO_IA,
            max_tokens=MAX_TOKENS,
            system=SISTEMA,
            # esfuerzo bajo: es una tarea de redacción corta a partir de datos ya resueltos,
            # no un problema que requiera razonar mucho -- sale más rápido y cuesta menos
            output_config={'effort': 'low'},
            messages=[{'role': 'user', 'content': _prompt(hechos)}],
        )
    except anthropic.AuthenticationError:
        return None, 'La clave de API no es válida.'
    except anthropic.RateLimitError:
        return None, 'Se alcanzó el límite de consultas de la API.'
    except anthropic.APIConnectionError:
        return None, 'No se pudo conectar con la API (sin red o servicio caído).'
    except anthropic.APIStatusError as e:
        return None, f'La API respondió con un error {e.status_code}.'
    except Exception as e:
        # cualquier otra cosa -- versión antigua de la librería que no acepta output_config,
        # un cambio de la API, lo que sea. La app no se cae por esto.
        return None, f'Fallo inesperado al generar el texto: {type(e).__name__}: {e}'

    if respuesta.stop_reason == 'refusal':
        return None, 'El modelo declinó responder.'
    if respuesta.stop_reason == 'max_tokens':
        return None, 'La respuesta quedó cortada.'

    texto = ' '.join(bloque.text for bloque in respuesta.content
                     if getattr(bloque, 'type', None) == 'text').strip()
    if not texto:
        return None, 'La respuesta vino vacía.'

    ok, intruso = _verificar_numeros(texto, hechos)
    if not ok:
        return None, f'Se descartó el texto: mencionaba el número {intruso}, que no está en los datos.'

    return texto, 'Texto redactado con IA y verificado contra los datos.'


# ── La función que usa la interfaz ────────────────────────────────────────────────

def explicar(hechos, usar_ia=True):
    """{'texto', 'origen': 'ia'|'plantilla', 'detalle'}.

    Siempre devuelve un texto utilizable. `usar_ia=False` fuerza la plantilla, que es útil
    para comparar las dos versiones o para presentar sin depender de la red."""
    if not usar_ia:
        return {'texto': texto_plantilla(hechos), 'origen': 'plantilla',
                'detalle': 'Explicación automática sin IA (desactivada).'}

    llave = repr(sorted(hechos.items(), key=lambda kv: kv[0]))
    if llave in _CACHE:
        return _CACHE[llave]

    texto, detalle = texto_ia(hechos)
    if texto:
        resultado = {'texto': texto, 'origen': 'ia', 'detalle': detalle}
    else:
        resultado = {'texto': texto_plantilla(hechos), 'origen': 'plantilla', 'detalle': detalle}

    if len(_CACHE) >= _CACHE_MAXIMO:
        _CACHE.clear()
    _CACHE[llave] = resultado
    return resultado
