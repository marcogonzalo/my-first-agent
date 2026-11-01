"""Paso 2: chat con memoria en la terminal.

Tú mantienes una lista de mensajes. Cada línea del usuario y cada respuesta
del asistente se añaden. Esa lista ES la memoria.
"""

import os

from common import MODEL, check_api_key_exists, load_env
from openai import OpenAI

# El modelo puede ignorar estas reglas. La barandilla real llega en el paso 6.
SYSTEM_PROMPT = """You are conversational agent running in the user's terminal.
Reply briefly, politely and precisely. Keep every question and answer short."""


def chat_with_memory() -> None:
    # Clave explícita: .env → os.environ → cliente.
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    print("""
    Este es un chat con memoria. Escribe 'salir' para salir.
    Cómo probar: di tu nombre, luego pregunta cuál es tu nombre.
    Recuerda durante la conversación.
    Si sales y pruebas de nuevo, olvidará.
    """)
    print("*" * 75)
    print()

    # Bucle que permite continuar la conversación hasta que se escriba 'salir'.
    while True:
        user_text = input("Tú: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"salir", "cerrar", "exit", "quit", "q"}:
            print()
            print("¡Adiós! ¡Vuelve pronto!")
            print()
            break

        messages.append({"role": "user", "content": user_text})

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            reasoning_effort="none",
        )
        message = response.choices[0].message
        print(f"{message.role.capitalize()}: {message.content or ''}")

        # MEMORIA: si comentas esta línea, no recuerda lo que se ha dicho.
        messages.append(message.model_dump(exclude_none=True))


# --- runner ---
# Arranque (entorno + clave). La lección está arriba.


def main() -> None:
    load_env()
    check_api_key_exists()
    chat_with_memory()


if __name__ == "__main__":
    main()
