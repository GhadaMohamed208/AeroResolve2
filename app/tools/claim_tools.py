"""Action tools used by the Claim and Communication agents (write to the database)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.database.db import ACTIVE_CLAIM_STATUSES, DatabaseError, get_conn
from app.tools.base import ToolResult, safe_tool

# Business rules live in the TOOL, not in the LLM: the tool re-reads the baggage
# record and refuses to create claims that the airline's rules do not allow.
CLAIM_TYPE_BY_STATUS = {
    "Delayed": "Delayed Baggage",
    "Not Found": "Lost Baggage Investigation",
    "Under Investigation": "Baggage Investigation",
}
INITIAL_CLAIM_STATUS = {
    "Delayed Baggage": "Open",
    "Lost Baggage Investigation": "Under Investigation",
    "Baggage Investigation": "Under Investigation",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _active_claim(conn, tag: str):
    marks = ",".join("?" for _ in ACTIVE_CLAIM_STATUSES)
    return conn.execute(
        f"SELECT * FROM claims WHERE baggage_tag = ? AND status IN ({marks}) ORDER BY seq DESC LIMIT 1",
        (tag, *ACTIVE_CLAIM_STATUSES)).fetchone()


@safe_tool
def get_existing_claim(db_path, baggage_tag) -> ToolResult:
    """Return the active claim for a tag, or ``claim=None``."""
    with get_conn(db_path) as conn:
        row = _active_claim(conn, str(baggage_tag).upper())
    return ToolResult(True, data={"claim": dict(row) if row else None})


@safe_tool
def create_baggage_claim(db_path, baggage_tag, booking_reference, passenger_name, flight_number,
                         flight_date, claim_type, contact_email=None) -> ToolResult:
    """Create and persist a baggage claim (idempotent per baggage tag)."""
    tag = str(baggage_tag).upper()
    with get_conn(db_path) as conn:
        bag = conn.execute("SELECT * FROM baggage WHERE baggage_tag = ?", (tag,)).fetchone()
        if bag is None:
            return ToolResult(False, error_code="TAG_NOT_FOUND", message=f"Unknown baggage tag {tag}.")
        if bag["booking_reference"].upper() != str(booking_reference).upper():
            return ToolResult(False, error_code="TAG_BOOKING_MISMATCH",
                              message="The baggage tag does not belong to that booking.")
        if bag["status"] == "Delivered":
            return ToolResult(False, error_code="BAGGAGE_ALREADY_DELIVERED",
                              message="Baggage is already delivered - no claim can be created.")
        expected_type = CLAIM_TYPE_BY_STATUS.get(bag["status"])
        if expected_type is None:
            return ToolResult(False, error_code="NOT_ELIGIBLE",
                              message=f"Baggage with status '{bag['status']}' does not require a claim.")
        if claim_type != expected_type:
            return ToolResult(False, error_code="INVALID_CLAIM_TYPE",
                              message=f"Claim type '{claim_type}' does not fit status '{bag['status']}'.")
        existing = _active_claim(conn, tag)
        if existing is not None:
            return ToolResult(True, data={"claim": dict(existing), "created": False, "duplicate": True},
                              message=f"An active claim ({existing['claim_id']}) already exists.")
        location = bag["last_known_location"]  # always taken from the database
        try:
            cur = conn.execute(
                "INSERT INTO claims (claim_id, baggage_tag, booking_reference, passenger_name,"
                " flight_number, flight_date, claim_type, status, last_known_location, contact_email,"
                " created_at) VALUES (NULL,?,?,?,?,?,?,?,?,?,?)",
                (tag, str(booking_reference).upper(), passenger_name, str(flight_number).upper(),
                 flight_date, claim_type, INITIAL_CLAIM_STATUS[claim_type], location, contact_email, _now()))
        except DatabaseError:
            raise
        claim_id = f"CLM-{1000 + cur.lastrowid}"
        conn.execute("UPDATE claims SET claim_id = ? WHERE seq = ?", (claim_id, cur.lastrowid))
        row = conn.execute("SELECT * FROM claims WHERE seq = ?", (cur.lastrowid,)).fetchone()
    return ToolResult(True, data={"claim": dict(row), "created": True, "duplicate": False})


@safe_tool
def create_follow_up(db_path, baggage_tag, booking_reference, follow_up_type, note,
                     claim_id=None, due_hours=24) -> ToolResult:
    """Create a follow-up record (monitoring task, status-update reminder or manual review)."""
    created = datetime.now(timezone.utc)
    due = (created + timedelta(hours=due_hours)).strftime("%Y-%m-%d %H:%M:%S")
    with get_conn(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO follow_ups (claim_id, baggage_tag, booking_reference, follow_up_type, due_at,"
            " status, note, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (claim_id, str(baggage_tag).upper(), str(booking_reference).upper(), follow_up_type, due,
             "Scheduled", note, created.strftime("%Y-%m-%d %H:%M:%S")))
        row = conn.execute("SELECT * FROM follow_ups WHERE id = ?", (cur.lastrowid,)).fetchone()
    return ToolResult(True, data=dict(row))


@safe_tool
def save_email(db_path, claim_id, baggage_tag, recipient, subject, body) -> ToolResult:
    """Persist the generated passenger email (prototype: stored as a preview, never sent)."""
    with get_conn(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO emails (claim_id, baggage_tag, recipient, subject, body, status, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (claim_id, baggage_tag, recipient, subject, body, "Preview (not sent)", _now()))
        row = conn.execute("SELECT * FROM emails WHERE id = ?", (cur.lastrowid,)).fetchone()
    return ToolResult(True, data=dict(row))


@safe_tool
def list_records(db_path, table, limit=200) -> ToolResult:
    """Read-only listing for the dashboard (whitelisted tables only)."""
    order = {"claims": "seq", "follow_ups": "id", "emails": "id"}
    if table not in order:
        return ToolResult(False, error_code="BAD_TABLE", message="Unknown table.")
    with get_conn(db_path) as conn:
        rows = conn.execute(f"SELECT * FROM {table} ORDER BY {order[table]} DESC LIMIT ?", (limit,)).fetchall()
    return ToolResult(True, data={"rows": [dict(r) for r in rows]})
