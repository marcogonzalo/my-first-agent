"""Paso 3: la primera herramienta (read_file).

El modelo no inventa el contenido del archivo. Pide llamar a read_file. Tú
ejecutas la función, devuelves el resultado con el mismo tool_call id
y vuelves a llamar.
"""

import json
import os
from pathlib import Path

from common import MODEL, check_api_key_exists, load_env
from openai import OpenAI

# El modelo puede ignorar estas reglas. La barandilla real llega en el paso 6.
SYSTEM_PROMPT = """You are an assistant agent running in the user's terminal.
You can read files with read_file to answer questions about file contents.
Paths are relative to the working directory, the folder the user launched you from.
Reply briefly. Be polite and precise. Keep every question and answer short.
"""

# Esquemas de las herramientas: describe las herramientas y sus parámetros.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path, relative to the working directory.",
                    },
                },
                "required": ["path"],
            },
        },
    },
]


# Herramienta para leer un archivo
def read_file(path: str) -> str:
    # Devolvemos el error como texto: si petamos, el agente muere.
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        return f"Error reading {path!r}: {exc}"


# Una vuelta manual: pregunta → tool_call → ejecutar → resultado → respuesta final.
def run_one_tool_round(client: OpenAI, messages: list[dict], user_text: str) -> None:
    # Una vuelta manual: pregunta → tool_call → ejecutar → resultado → respuesta final.
    # Aquí se ve el id de la herramienta y por qué hay que guardarlo en el historial.
    print("============================ MESSAGE ============================")
    print()
    print("User:", user_text)
    messages.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_SCHEMAS,
        reasoning_effort="none",
    )
    choice = response.choices[0]
    message = choice.message

    print(f"(finish_reason: {choice.finish_reason})")
    print()
    print("------------ ANSWER OBJECT ---------")
    print(message.model_dump())
    print("------------------------------------")

    # OBLIGATORIO: guardar la petición de herramienta ANTES del resultado.
    # Si te saltas esto, la API rechaza el resultado (error más común).
    messages.append(message.model_dump(exclude_none=True))

    if not message.tool_calls:
        print("No tool call. Assistant:", message.content)
        return

    # Llamada a la herramienta
    for tool_call in message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments or "{}")
        print()
        print(f"- Tool calling: {name}({args})")
        if name != "read_file":
            result = f"Unknown tool: {name}"
        else:
            result = read_file(**args)

        print()
        print("------------ TOOL RESULT ---------")
        print(result)
        print("----------------------------------")
        print()

        # Agregar resultado de la herramienta al historial de mensajes
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )

    # Generar respuesta del modelo con mensajes actualizados
    follow_up = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_SCHEMAS,
        reasoning_effort="none",
    )
    final_message = follow_up.choices[0].message
    messages.append(final_message.model_dump(exclude_none=True))

    print(f"Agent: {final_message.content}")
    print()
    print("================================================================= ")
    print()


# Demo de la herramienta read_file
def demo_read_file() -> None:
    # Clave explícita: .env → os.environ → cliente.
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Caso feliz: el modelo pide read_file, leemos y responde con el contenido.
    run_one_tool_round(client, messages, "What is in workspace/notes.txt?")
    # Caso error: archivo ausente; error como texto para que el modelo reaccione.
    run_one_tool_round(client, messages, "What is in workspace/missing.txt?")


# --- runner ---
# Arranque (entorno + clave). La lección está arriba.


def main() -> None:
    load_env()
    check_api_key_exists()
    demo_read_file()


if __name__ == "__main__":
    main()
