"""Paso 1: una pregunta, una respuesta.

Esto AÚN NO es un agente. Creas el cliente, eliges el modelo, envías un solo
mensaje de usuario e inspeccionas la respuesta completa (rol, tokens, finish_reason).
"""

import os

from common import MODEL, check_api_key_exists, load_env
from openai import OpenAI


def test_model() -> None:
    # Clave explícita: .env → os.environ → cliente.
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = [
        {"role": "user", "content": "En una frase, ¿qué es un agente de código?"},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        reasoning_effort="none",
    )

    # Imprimimos la respuesta COMPLETA:
    # metadatos, tokens y motivo de parada para explorar la respuesta.
    print("--------- ANSWER OBJECT ---------")
    print(response.model_dump())
    print()
    print("------------ TOKENS -------------")
    print("Total:", response.usage.total_tokens)
    print("Input (prompt):", response.usage.prompt_tokens)
    print("Output (completion):", response.usage.completion_tokens)
    print()
    print("------------ ANSWER -------------")
    print("Assistant:", response.choices[0].message.content)
    print()


# --- runner ---
# Arranque (entorno + clave). La lección está arriba.


def main() -> None:
    load_env()
    check_api_key_exists()
    test_model()


if __name__ == "__main__":
    main()
