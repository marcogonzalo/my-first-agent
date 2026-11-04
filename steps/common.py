"""Utilidades compartidas: entorno y clave. No es la lección."""

import os
import sys
from pathlib import Path

# Mismo modelo en todos los pasos (Chat Completions + tools).
MODEL = "gpt-5.4-mini"

# Raíz del repo (padre de steps/).
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path | None = None) -> None:
    # Carga KEY=VALUE desde .env sin pisar variables ya definidas.
    env_path = path or REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def check_api_key_exists() -> None:
    # Verifica que existe la clave de la API
    # Sale con mensaje claro si no existe la clave.
    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY. Copy .env.example to .env and set the key.")
        sys.exit(1)
