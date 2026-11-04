"""Agente de programación con herramientas y barandillas.

Herramientas: list_files, read_file, write_file, run_command.
write_file y run_command piden un "ok" explícito antes de ejecutarse.
Las rutas deben quedarse dentro del directorio de lanzamiento. Los comandos tienen timeout.
"""

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openai import OpenAI

from steps.common import MODEL, load_env, check_api_key_exists

# Tope de vueltas del bucle: evita un error infinito y una factura inesperada.
MAX_ITERATIONS = 15
# Tope de salida de herramienta: un archivo o comando grande no debe llenar el contexto.
MAX_TOOL_OUTPUT = 4000
# Tiempo máximo por comando: un proceso colgado no bloquea el agente para siempre.
COMMAND_TIMEOUT_SECONDS = 10

# El modelo puede ignorar la frase de confirmación. La GUARDRAIL real está abajo.
SYSTEM_PROMPT = """You are a coding agent running in the user's terminal.
You can list files, read files, write files, and run shell commands.
Use your tools to complete the user's task, then briefly summarize what you did.
Ask for confirmation before modifying files.
The working directory is the folder the user launched you from.
Be polite and precise. Keep every question and answer short.
Don't repeat yourself in the same message.
"""

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
            "description": "Write text to a file. The user approves it first.",
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
                "Run a shell command and return its output. The user approves it first."
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

# Capturado al arrancar: el prompt dice "la carpeta desde la que lanzaste".
WORK_ROOT = Path.cwd().resolve()

ApproveFn = Callable[[str], bool]


# GUARDRAIL: jaula de rutas (realpath) + rechazo de .env.
def resolve_in_work_root(path: str) -> Path | str:
    """Resuelve la ruta y la mantiene dentro de WORK_ROOT.

    Devuelve texto de error si falla.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = WORK_ROOT / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:
        return f"GUARDRAIL: Error resolviendo {path!r}: {exc}"

    try:
        resolved.relative_to(WORK_ROOT)
    except ValueError:
        return f"GUARDRAIL: Ruta sale del directorio de trabajo: {path!r}"

    if resolved.name == ".env" or resolved == WORK_ROOT / ".env":
        return "GUARDRAIL: Rechazado: .env no es accesible a través de las herramientas de archivo."

    return resolved


# Lista archivos; pasa por resolve_in_work_root.
def list_files(path: str) -> str:
    resolved = resolve_in_work_root(path)
    if isinstance(resolved, str):
        return resolved
    try:
        if not resolved.exists():
            return f"Error listing {path!r}: path does not exist"
        if not resolved.is_dir():
            return f"Error listing {path!r}: not a directory"
        entries = sorted(resolved.iterdir(), key=lambda p: p.name.lower())
        if not entries:
            return "(empty directory)"
        lines = []
        for entry in entries:
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{entry.name}{suffix}")
        return "\n".join(lines)
    except OSError as exc:
        return f"Error listing {path!r}: {exc}"


# Lee un archivo; pasa por resolve_in_work_root (y bloquea .env).
def read_file(path: str) -> str:
    resolved = resolve_in_work_root(path)
    if isinstance(resolved, str):
        return resolved
    try:
        return resolved.read_text(encoding="utf-8")
    except OSError as exc:
        return f"GUARDRAIL: Error leyendo {path!r}: {exc}"


# Escribe un archivo; jaula de rutas. La aprobación "ok" está en execute_tool.
def write_file(path: str, content: str) -> str:
    resolved = resolve_in_work_root(path)
    if isinstance(resolved, str):
        return resolved
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        # Revalidar el padre tras mkdir (trucos con enlaces simbólicos).
        parent = resolved.parent.resolve()
        try:
            parent.relative_to(WORK_ROOT)
        except ValueError:
            return f"GUARDRAIL: Ruta sale del directorio de trabajo: {path!r}"
        resolved.write_text(content, encoding="utf-8")
        return f"GUARDRAIL: Escrito {len(content)} caracteres a {path}"
    except OSError as exc:
        return f"GUARDRAIL: Error escribiendo {path!r}: {exc}"


# Ejecuta un comando con timeout y sin OPENAI_API_KEY en el entorno.
# GUARDRAIL parcial: el shell aún puede escapar (p. ej. cat .env); el "ok" es la otra.
def run_command(command: str) -> str:
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(WORK_ROOT),
            timeout=COMMAND_TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return (
            f"Error: command timed out after {COMMAND_TIMEOUT_SECONDS} seconds: "
            f"{command!r}"
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
TOOLS: dict[str, Callable[..., str]] = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
}

# Tools que exigen "ok" explícito antes de ejecutarse.
DANGEROUS_TOOLS = frozenset({"write_file", "run_command"})


# GUARDRAIL: recorta salidas largas para no hinchar el contexto.
def truncate_output(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    print("GUARDRAIL: truncando salida larga")
    return text[:MAX_TOOL_OUTPUT] + "\n...[output truncated]"


# GUARDRAIL: pide "ok" en la terminal antes de write/run.
def default_approve(action: str) -> bool:
    print(action)
    try:
        print()
        print("- GUARDRAIL: preguntar por aprobación. -")
        answer = input("Agente: ¿Puedo hacer esto? Escribe 'ok' para aprobar: ")
    except EOFError, KeyboardInterrupt:
        print()
        return False
    return answer.strip().lower() == "ok"


def describe_action(name: str, arguments: dict) -> str:
    if name == "write_file":
        path = arguments.get("path", "?")
        content = arguments.get("content", "")
        preview = content if len(content) <= 120 else content[:120] + "..."
        return f"write_file ruta={path!r} contenido={preview!r}"
    if name == "run_command":
        return f"run_command comando={arguments.get('command', '?')!r}"
    return f"{name}({arguments})"


# GUARDRAIL + despacho: aprueba tools peligrosas, llama la función, recorta salida.
def execute_tool(
    name: str,
    arguments: dict,
    approve: ApproveFn | None = None,
) -> str:
    print(f"→ Tool call: {name}({arguments})")
    fn = TOOLS.get(name)
    if fn is None:
        return f"Unknown tool: {name}"

    if name in DANGEROUS_TOOLS:
        ask = approve or default_approve
        if not ask(describe_action(name, arguments)):
            return "GUARDRAIL: Usuario rechazó esta acción."

    try:
        result = fn(**arguments)
    except TypeError as exc:
        result = f"GUARDRAIL: Argumentos inválidos para {name}: {exc}"
    except Exception as exc:  # noqa: BLE001 — el agente debe seguir vivo
        result = f"GUARDRAIL: Error en {name}: {exc}"
    return truncate_output(str(result))


# --- Bucle del agente (igual que el paso 4/5; las GUARDRAILs están en execute_tool) ---
def run_agent(
    client: Any,
    messages: list[dict],
    *,
    approve: ApproveFn | None = None,
    model: str = MODEL,
    max_iterations: int = MAX_ITERATIONS,
) -> str:
    """Llama al modelo, ejecuta tools, añade resultados y repite hasta terminar."""
    for iteration in range(1, max_iterations + 1):
        print(f"--- Iteration {iteration} ---")
        response = client.chat.completions.create(
            model=model,
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
            # Un mensaje puede pedir varias tools; un mensaje tool por cada id.
            for tool_call in message.tool_calls or []:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    result = f"JSON inválido: {exc}"
                else:
                    result = execute_tool(name, args, approve=approve)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
            continue

        if finish_reason == "length":
            note = "Parado: el modelo alcanzó el límite de tokens (finish_reason=length)."
            print(note)
            return note

        if finish_reason == "stop":
            return message.content or ""

        note = f"Parado: motivo de fin inesperado={finish_reason!r}."
        print(note)
        return note

    note = f"Parado: se alcanzó el tope de iteraciones ({max_iterations})."
    print(note)
    return note


# --- Chat interactivo con GUARDRAILs ---
def chat_with_guardrails() -> None:
    # Clave explícita: .env → os.environ → cliente.
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("================================================================")
    print("********************* Agente de código *************************")
    print("================================================================")
    print()
    print(f"- Directorio de trabajo: {WORK_ROOT}")
    print("- Escribe 'salir' para terminar.")
    while True:
        try:
            print()
            user_text = input("Tú: ").strip()
        except EOFError, KeyboardInterrupt:
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in {"salir", "cerrar", "quit", "exit", "q"}:
            break

        print()
        print("========================== AGENT TURN ==========================")
        print()

        messages.append({"role": "user", "content": user_text})
        answer = run_agent(client, messages)
        print()
        print("================================================================")
        print()
        print("Agente:", answer)


# --- runner ---
# Arranque (entorno + clave). La lección está arriba.


def main() -> None:
    load_env()
    check_api_key_exists()
    chat_with_guardrails()


if __name__ == "__main__":
    main()
