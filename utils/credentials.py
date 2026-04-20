import os
from pathlib import Path
from typing import Optional


def get_do_model_key() -> Optional[str]:
    """Return DO_MODEL_ACCESS_KEY from the environment or a top-level .env file.

    Looks first in environment variables, then in a `.env` file at the
    repository root (one level above `utils/`). Returns `None` if not found.
    """
    name = "DO_MODEL_ACCESS_KEY"
    key = os.environ.get(name)
    if key:
        return key

    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(name + "="):
                    _, val = line.split("=", 1)
                    return val.strip().strip('"').strip("'")
        except Exception:
            pass

    return None


def mask_key(k: str) -> str:
    if not k:
        return ""
    if len(k) <= 8:
        return "*" * len(k)
    return k[:4] + "*" * (len(k) - 8) + k[-4:]
