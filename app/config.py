"""Central configuration. Everything is driven by environment variables.

Order of precedence: real environment variables > ``.env`` file > Streamlit
secrets (used on Streamlit Community Cloud) > defaults. No secret is ever
hard-coded in the repository.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_ENV_LOADED = False


def _load_env_file() -> None:
    """Minimal .env loader (avoids an extra dependency)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _streamlit_secret(name: str):
    st = sys.modules.get("streamlit")
    if st is None:
        return None
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets file configured
        return None


def get_setting(name: str, default: str = "") -> str:
    _load_env_file()
    value = os.environ.get(name)
    if value not in (None, ""):
        return value
    secret = _streamlit_secret(name)
    if secret not in (None, ""):
        return str(secret)
    return default


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_timeout: float
    ollama_base_url: str
    ollama_model: str
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    anthropic_api_key: str
    anthropic_model: str
    db_path: str
    data_dir: str
    policy_pdf: str
    exports_dir: str

    @classmethod
    def load(cls) -> "Settings":
        data_dir = get_setting("AERORESOLVE_DATA_DIR", str(ROOT / "data"))
        return cls(
            llm_provider=get_setting("LLM_PROVIDER", "rules").lower(),
            llm_timeout=float(get_setting("LLM_TIMEOUT", "30")),
            ollama_base_url=get_setting("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            ollama_model=get_setting("OLLAMA_MODEL", "llama3.2"),
            openai_base_url=get_setting("OPENAI_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
            openai_api_key=get_setting("OPENAI_API_KEY", ""),
            openai_model=get_setting("OPENAI_MODEL", "llama-3.1-8b-instant"),
            anthropic_api_key=get_setting("ANTHROPIC_API_KEY", ""),
            anthropic_model=get_setting("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            db_path=get_setting("AERORESOLVE_DB_PATH", str(Path(data_dir) / "aeroresolve.db")),
            data_dir=data_dir,
            policy_pdf=get_setting("AERORESOLVE_POLICY_PDF", str(Path(data_dir) / "baggage_policy.pdf")),
            exports_dir=get_setting("AERORESOLVE_EXPORTS_DIR", str(Path(data_dir) / "exports")),
        )
