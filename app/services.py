"""Dependency container: database path, LLM client, policy PDF and export directory."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.database.db import init_db
from app.llm.client import LLMClient, create_llm


@dataclass
class Services:
    db_path: str
    llm: LLMClient
    policy_path: str
    exports_dir: str
    data_dir: str


def _writable_dir(preferred: str) -> str:
    try:
        Path(preferred).mkdir(parents=True, exist_ok=True)
        probe = Path(preferred) / ".probe"
        probe.write_text("ok")
        probe.unlink()
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "aeroresolve_exports"
        fallback.mkdir(parents=True, exist_ok=True)
        return str(fallback)


def build_services(db_path: str | None = None, llm: LLMClient | None = None,
                   settings: Settings | None = None) -> Services:
    s = settings or Settings.load()
    used_db = init_db(db_path or s.db_path, s.data_dir)
    return Services(db_path=used_db, llm=llm or create_llm(s), policy_path=s.policy_pdf,
                    exports_dir=_writable_dir(s.exports_dir), data_dir=s.data_dir)
