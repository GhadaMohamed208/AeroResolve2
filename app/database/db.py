"""SQLite persistence layer (flights, baggage, claims, follow-ups, emails)."""
from __future__ import annotations

import csv
import logging
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.config import ROOT

log = logging.getLogger(__name__)

BAGGAGE_STATUSES = (
    "Delivered", "Delayed", "In Transit", "At Airport", "Not Found", "Under Investigation",
)
ACTIVE_CLAIM_STATUSES = ("Open", "Under Investigation")


class DatabaseError(Exception):
    """Raised for any low-level database failure (wraps sqlite3.Error)."""


_STATUS_SQL = ", ".join(f"'{s}'" for s in BAGGAGE_STATUSES)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS flights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flight_number TEXT NOT NULL,
    flight_date TEXT NOT NULL,
    origin TEXT NOT NULL,
    destination TEXT NOT NULL,
    booking_reference TEXT NOT NULL UNIQUE,
    passenger_name TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_flights_flight ON flights(flight_number, flight_date);

CREATE TABLE IF NOT EXISTS baggage (
    baggage_tag TEXT PRIMARY KEY,
    booking_reference TEXT NOT NULL REFERENCES flights(booking_reference),
    flight_number TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ({_STATUS_SQL})),
    last_known_location TEXT,
    last_updated TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id TEXT UNIQUE,
    baggage_tag TEXT NOT NULL REFERENCES baggage(baggage_tag),
    booking_reference TEXT NOT NULL,
    passenger_name TEXT NOT NULL,
    flight_number TEXT NOT NULL,
    flight_date TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    status TEXT NOT NULL,
    last_known_location TEXT,
    contact_email TEXT,
    created_at TEXT NOT NULL
);
-- Database-level duplicate protection: only ONE active claim per baggage tag.
CREATE UNIQUE INDEX IF NOT EXISTS ux_claims_active_tag
    ON claims(baggage_tag) WHERE status IN ('Open', 'Under Investigation');

CREATE TABLE IF NOT EXISTS follow_ups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id TEXT,
    baggage_tag TEXT NOT NULL,
    booking_reference TEXT NOT NULL,
    follow_up_type TEXT NOT NULL,
    due_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'Scheduled',
    note TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id TEXT,
    baggage_tag TEXT,
    recipient TEXT NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'Preview (not sent)',
    created_at TEXT NOT NULL
);
"""

_TABLES = ("emails", "follow_ups", "claims", "baggage", "flights")


@contextmanager
def get_conn(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open a connection; commit on success, roll back on error."""
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
    except sqlite3.Error as exc:
        raise DatabaseError(f"cannot open database: {exc}") from exc
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise DatabaseError(str(exc)) from exc
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def seed_from_csv(db_path: str | Path, data_dir: str | Path | None = None) -> None:
    """Load flights.csv, baggage.csv and claims.csv into an empty database."""
    data_dir = Path(data_dir or ROOT / "data")
    flights = _read_csv(data_dir / "flights.csv")
    baggage = _read_csv(data_dir / "baggage.csv")
    claims = _read_csv(data_dir / "claims.csv")
    with get_conn(db_path) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO flights (flight_number, flight_date, origin, destination,"
            " booking_reference, passenger_name) VALUES (:flight_number, :flight_date, :origin,"
            " :destination, :booking_reference, :passenger_name)", flights)
        conn.executemany(
            "INSERT OR IGNORE INTO baggage (baggage_tag, booking_reference, flight_number, status,"
            " last_known_location, last_updated) VALUES (:baggage_tag, :booking_reference,"
            " :flight_number, :status, :last_known_location, :last_updated)", baggage)
        for c in claims:
            seq = int(c["claim_id"].split("-")[1]) - 1000
            conn.execute(
                "INSERT OR IGNORE INTO claims (seq, claim_id, baggage_tag, booking_reference,"
                " passenger_name, flight_number, flight_date, claim_type, status,"
                " last_known_location, contact_email, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (seq, c["claim_id"], c["baggage_tag"], c["booking_reference"], c["passenger_name"],
                 c["flight_number"], c["flight_date"], c["claim_type"], c["status"],
                 c["last_known_location"], c["contact_email"], c["created_at"]))


def _init(db_path: str | Path, data_dir: str | Path | None) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)
        empty = conn.execute("SELECT COUNT(*) FROM flights").fetchone()[0] == 0
    if empty:
        seed_from_csv(db_path, data_dir)


def init_db(db_path: str | Path, data_dir: str | Path | None = None) -> str:
    """Create the schema and seed it if empty. Returns the path actually used.

    If the requested location is not writable (some hosts have read-only
    checkouts) we transparently fall back to a temp directory.
    """
    try:
        _init(db_path, data_dir)
        return str(db_path)
    except (OSError, DatabaseError) as exc:
        fallback = Path(tempfile.gettempdir()) / "aeroresolve" / "aeroresolve.db"
        log.warning("DB path %s unusable (%s); falling back to %s", db_path, exc, fallback)
        _init(fallback, data_dir)
        return str(fallback)


def reset_db(db_path: str | Path, data_dir: str | Path | None = None) -> None:
    """Drop all data and re-seed from the CSV files (used by the demo UI)."""
    with get_conn(db_path) as conn:
        for table in _TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute("DROP INDEX IF EXISTS ux_claims_active_tag")
    _init(db_path, data_dir)
