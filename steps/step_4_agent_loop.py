"""Paso 4: el bucle del agente (run_agent).

Llamas al modelo, ejecutas cada herramienta que pida, añades resultados
y vuelves a llamar. Paras cuando finish_reason == stop. Limitamos
iteraciones para que un bug no buclé sin fin.
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


# Llamada a la herramienta
def call_tool(name: str, arguments: dict) -> str:
    if name == "read_file":
        return read_file(**arguments)
    return f"Unknown tool: {name}"


# Repite: llamar al modelo → ejecutar tools → añadir resultados → volver a llamar.
def run_agent(client: OpenAI, messages: list[dict]) -> str:
    iteration = 0
    while True:
        iteration += 1
        print(f"--- Iteration {iteration} ---")
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            reasoning_effort="none",
        )
        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason
        print(f"(finish_reason: {finish_reason})")
        print()

        # Guardar SIEMPRE el turno del asistente (con tool_calls si los hay).
        messages.append(message.model_dump(exclude_none=True))

        if finish_reason == "tool_calls":
            # Un solo mensaje del modelo puede pedir VARIAS herramientas.
            # OpenAI exige un mensaje tool por cada id (no un solo mensaje juntos).
            for tool_call in message.tool_calls or []:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    result = f"Invalid JSON arguments: {exc}"
                else:
                    print(f"- Tool calling: {name}({args})")
                    result = call_tool(name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
            continue

        if finish_reason == "length":
            # Cortado por tokens: NO ha terminado aunque parezca que sí.
            note = "Stopped: the model hit the token limit (finish_reason=length)."
            print(note)
            return note

        if finish_reason == "stop":
            return message.content or ""

        note = f"Stopped: unexpected finish_reason={finish_reason!r}."
        print(note)
        return note


def demo_three_files() -> None:
    # Clave explícita: .env → os.environ → cliente.
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Read workspace/uno.txt, workspace/dos.txt, and "
                "workspace/tres.txt. Reply with their contents in one short line."
            ),
        },
    ]
    answer = run_agent(client, messages)
    print("Agent:", answer)


# --- runner ---
# Arranque (entorno + clave). La lección está arriba.


def main() -> None:
    load_env()
    check_api_key_exists()
    demo_three_files()


if __name__ == "__main__":
    main()
