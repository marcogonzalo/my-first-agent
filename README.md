# Mi primer agente

Un agente de programación desde cero en Python puro. Sin frameworks. Solo OpenAI Chat Completions.

Aprendes una idea por paso: una llamada a la API, luego memoria, luego una herramienta, luego el bucle del agente, luego la caja completa y, al final, las barandillas.

## Paso 0: preparar el entorno

Este proyecto necesita **Python 3.15.2** y **uv**.

```bash
# Instala uv si aún no lo tienes:
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Preferible: CPython 3.15.2 oficial cuando uv pueda descargarlo:
uv python install 3.15.2
uv python pin 3.15.2

# Si uv aún no descarga 3.15.2, compílalo desde python.org y apunta uv a ese intérprete:
# ./configure --prefix=$HOME/.local/python-3.15.2-build --with-openssl="$(brew --prefix openssl@3)"
# make -j && make install
# uv venv --python $HOME/.local/python-3.15.2-build/bin/python3.15

uv sync
cp .env.example .env
# Pon tu clave de OpenAI en .env. Nunca subas .env a Git.
```

Lanza siempre los comandos desde la **raíz del repositorio**. Esa carpeta es el directorio de trabajo del agente.

Cada paso es **un solo archivo**. Arriba está la lección (incluye `api_key = os.environ["OPENAI_API_KEY"]` y `client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])`); abajo, tras `# --- runner ---`, solo arranque (`.env` + comprobar clave). Lo repetido vive en [`steps/common.py`](steps/common.py) (`load_env`, `check_api_key_exists`, `MODEL`).

## Ejecutar los pasos

```bash
uv run python steps/step_1_first_call.py
uv run python steps/step_2_memory.py
uv run python steps/step_3_first_tool.py
uv run python steps/step_4_agent_loop.py
uv run python steps/step_5_toolbox.py
uv run python steps/step_6_guardrails.py
uv run python agent.py
```

## Qué muestra cada paso

### Paso 1 — primera llamada

Un solo mensaje de usuario. El script imprime la respuesta **completa** de la API (`role`, `usage`, `finish_reason`), no solo el texto.

Esto todavía no es un agente. Es una pregunta y una respuesta.

### Paso 2 — memoria

Un chat en la terminal. Tu línea entra en `messages`. La respuesta del modelo también.

Demo: dile tu nombre y luego pregunta cómo te llamas. Lo recuerda.

Luego rómpelo: comenta la línea que guarda la respuesta del asistente (marcada con un comentario `#`). Pregunta otra vez. Ya no lo sabe. La memoria es la lista que **tú** gestionas.

Un mensaje de sistema fija las reglas. El modelo puede ignorarlas. Las barandillas en código llegan después.

### Paso 3 — primera herramienta

`read_file` y su esquema JSON. Pregunta: ¿qué hay en `workspace/notes.txt`?

El modelo no contesta con texto final: pide una **llamada a herramienta** (nombre, argumentos, id). Debes:

1. Guardar ese mensaje del asistente (con la petición) en el historial.
2. Ejecutar la función.
3. Añadir un mensaje `role: "tool"` con el **mismo** id.
4. Volver a llamar al modelo.

Saltarte el paso 1 es el error más común el primer día.

Prueba también un archivo que no existe. Si tu código lanza una excepción, el agente muere. Si devuelves el error como texto, el modelo puede reaccionar.

### Paso 4 — bucle del agente

`run_agent` repite: llamar → ¿hay herramientas? → ejecutarlas → volver a llamar.

Sale cuando `finish_reason` es `stop`. Si es `length`, el modelo se cortó por tokens y **no** ha terminado.

Tope de 15 llamadas al modelo para que un bug no gaste tokens sin fin.

La demo lee tres archivos a la vez. Una respuesta puede pedir varias herramientas. Tu código debe ejecutarlas **todas** y devolver **un mensaje tool por cada id**.

### Paso 5 — caja de herramientas completa

`list_files`, `read_file`, `write_file`, `run_command`.

Un diccionario `TOOLS` conecta nombre → función. `execute_tool` imprime la llamada, captura errores como texto y recorta salidas largas.

**Aviso:** este paso puede escribir y ejecutar **sin** pedir confirmación.

### Paso 6 / `agent.py` — barandillas

El mismo chat, más:

- `write_file` y `run_command` preguntan `May I do this? Type yes to approve:`
- Las rutas deben quedar dentro del directorio de lanzamiento (`realpath`, no solo el texto)
- `.env` no se puede leer ni escribir con las herramientas de archivos
- Los comandos tienen timeout de 10 s y un entorno sin `OPENAI_API_KEY`

El system prompt sigue pidiendo confirmación. Eso no es una barandilla. La barandilla real está en tu código.

## Tests

No hace falta clave de API:

```bash
uv run python -m unittest discover -s tests -v
```

## Modelo

`gpt-5.4-mini` con `reasoning_effort="none"` para que el tool calling funcione en Chat Completions.
