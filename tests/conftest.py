"""Shared pytest fixtures.

Every test gets its own throwaway SQLite database, seeded from the same
CSV fixtures used by the app, so tests never touch the real demo database
and never depend on run order.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.llm.client import LLMClient, RulesProvider  # noqa: E402
from app.services import Services, build_services  # noqa: E402
from app.workflow.orchestrator import Orchestrator  # noqa: E402


@pytest.fixture()
def db_path(tmp_path) -> str:
    return str(tmp_path / f"test_{uuid.uuid4().hex}.db")


@pytest.fixture()
def services(db_path) -> Services:
    """Rule-based services (no network / no API key needed) against a fresh DB."""
    settings = Settings.load()
    return build_services(db_path=db_path, llm=LLMClient(RulesProvider()), settings=settings)


@pytest.fixture()
def orchestrator(services) -> Orchestrator:
    return Orchestrator(services)


@pytest.fixture()
def new_case(orchestrator):
    return orchestrator.new_case()
