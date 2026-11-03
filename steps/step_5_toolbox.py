"""Paso 5: la caja de herramientas completa.

Cuatro herramientas: list_files, read_file, write_file, run_command.
Un diccionario TOOLS conecta nombres con funciones. call imprime, captura y recorta.

AVISO: este paso puede escribir archivos y ejecutar comandos SIN pedir confirmación.
"""

import json
import os
import subprocess
from pathlib import Path

from common import MODEL, check_api_key_exists, load_env
from openai import OpenAI

# System prompt: describe al agente y sus herramientas.
# El modelo puede ignorar estas reglas. La barandilla real llega en el paso 6.
SYSTEM_PROMPT = """You are a coding agent running in the user's terminal.
You can list files, read files, write files, and run shell commands.
Use your tools to complete the user's task, then briefly summarize what you did.
The working directory is the folder the user launched you from.
Reply briefly. Be polite and precise. Keep every question and answer short."""

# Esquemas de las herramientas: describe las herramientas y sus parámetros.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and folders in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Directory to list, relative to the working directory. "
                            "Use . for the working directory."
                        ),
                    },
                },
                "required": ["path"],
            },
        },
    },
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
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path, relative to the working directory.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command and return its output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


# Herramienta para listar archivos
def list_files(path: str) -> str:
    try:
        entries = sorted(Path(path).iterdir(), key=lambda p: p.name.lower())
        if not entries:
            return "(empty directory)"
        lines = []
        for entry in entries:
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{entry.name}{suffix}")
        return "\n".join(lines)
    except OSError as exc:
        return f"Error listing {path!r}: {exc}"


# Herramienta para leer un archivo
def read_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        return f"Error reading {path!r}: {exc}"


# Herramienta para escribir un archivo
def write_file(path: str, content: str) -> str:
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path}"
    except OSError as exc:
        return f"Error writing {path!r}: {exc}"


# Herramienta para ejecutar un comando
def run_command(command: str) -> str:
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return f"Error running command: {exc}"
    parts = [
        f"exit_code={completed.returncode}",
        f"stdout:\n{completed.stdout}",
        f"stderr:\n{completed.stderr}",
    ]
    return "\n".join(parts)


# Diccionario nombre → función: evita un árbol de if/elif.
TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
}


# Llama a la herramienta
def call_tool(name: str, arguments: dict) -> str:
    print(f"- Tool calling: {name}({arguments})")
    fn = TOOLS.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    try:
        result = fn(**arguments)
    except TypeError as exc:
        result = f"Bad arguments for {name}: {exc}"
    except Exception as exc:  # noqa: BLE001 — el agente debe seguir vivo
        result = f"Error in {name}: {exc}"
    return str(result)


# Ejecuta el bucle agente:
# llama al modelo → ejecuta tools → añade resultados → vuelve a llamar → envía.
def run_agent(client: OpenAI, messages: list[dict]) -> str:
    print()
    print("Running agent loop...")
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
            # Un mensaje puede pedir VARIAS tools; un mensaje tool por cada id.
            for tool_call in message.tool_calls or []:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    result = f"Invalid JSON arguments: {exc}"
                else:
                    result = call_tool(name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

            # Reinicia el ciclo. Envía resultados al modelo y pide respuesta al agente.
            # (La siguiente vuelta del bucle reusa los mensajes con esos role:"tool".)
            print("→ Sending tool results to the model for the agent answer")
            continue

        if finish_reason == "length":
            # Cortado por tokens: NO ha terminado aunque parezca que sí.
            note = "Stopped: the model hit the token limit (finish_reason=length)."
            print(note)
            return note

        if finish_reason == "stop":
            # Respuesta final del agente (texto para el usuario).
            return message.content or ""

        note = f"Stopped: unexpected finish_reason={finish_reason!r}."
        print(note)
        return note


# Chat con la caja de herramientas
def chat_with_toolbox() -> None:
    # Clave explícita: .env → os.environ → cliente.
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    print()
    print("Chat con llamadas a herramientas (las usa sin pedir aprobación).")
    print("Escribe 'salir' para terminar.")
    print("-" * 75)
    print()

    while True:
        try:
            user_text = input("Tú: ").strip()
        except EOFError, KeyboardInterrupt:
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in {"salir", "cerrar", "exit", "quit", "q"}:
            break

        messages.append({"role": "user", "content": user_text})
        answer = run_agent(client, messages)
        print("Agente:", answer)


# --- runner ---
# Arranque (entorno + clave). La lección está arriba.


def main() -> None:
    load_env()
    check_api_key_exists()
    chat_with_toolbox()


if __name__ == "__main__":
    main()
