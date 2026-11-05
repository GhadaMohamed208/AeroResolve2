"""Retrieval tools used by the Baggage Tracking Agent (read-only)."""
from __future__ import annotations

import re

from app.database.db import get_conn
from app.tools.base import ToolResult, safe_tool


def _tokens(name: str) -> set[str]:
    return {t for t in re.sub(r"[^a-z\s]", " ", (name or "").casefold()).split() if t}


def names_match(given: str, actual: str) -> bool:
    """Tolerant name comparison: order/case/middle names don't matter."""
    g, a = _tokens(given), _tokens(actual)
    return bool(g) and bool(a) and (g <= a or a <= g)


@safe_tool
def check_booking(db_path, booking_reference, passenger_name=None, flight_number=None,
                  flight_date=None) -> ToolResult:
    """Validate that a booking exists and (optionally) matches name/flight/date."""
    ref = str(booking_reference).strip().upper()
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM flights WHERE booking_reference = ?", (ref,)).fetchone()
    if row is None:
        return ToolResult(False, error_code="BOOKING_NOT_FOUND", data={"field": "booking_reference"},
                          message=f"No booking was found with reference {ref}.")
    if passenger_name and not names_match(passenger_name, row["passenger_name"]):
        return ToolResult(False, error_code="NAME_MISMATCH", data={"field": "passenger_name"},
                          message=f"The passenger name does not match booking {ref}.")
    if flight_number and row["flight_number"].upper() != str(flight_number).upper():
        return ToolResult(False, error_code="FLIGHT_MISMATCH", data={"field": "flight_number"},
                          message=f"Flight {str(flight_number).upper()} does not match booking {ref}.")
    if flight_date and row["flight_date"] != flight_date:
        return ToolResult(False, error_code="DATE_MISMATCH", data={"field": "flight_date"},
                          message=f"The flight date {flight_date} does not match booking {ref}.")
    return ToolResult(True, data=dict(row))


@safe_tool
def get_flight_information(db_path, flight_number, flight_date) -> ToolResult:
    """Look up route information for a flight on a date."""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT flight_number, flight_date, origin, destination FROM flights"
            " WHERE flight_number = ? AND flight_date = ?", (str(flight_number).upper(), flight_date)
        ).fetchall()
    if not rows:
        return ToolResult(False, error_code="FLIGHT_NOT_FOUND", data={"field": "flight_number"},
                          message=f"No flight {str(flight_number).upper()} operates on {flight_date}.")
    info = dict(rows[0])
    info["passengers_on_record"] = len(rows)
    return ToolResult(True, data=info)


@safe_tool
def track_baggage(db_path, baggage_tag) -> ToolResult:
    """Return the current tracing record of a baggage tag."""
    tag = str(baggage_tag).strip().upper()
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM baggage WHERE baggage_tag = ?", (tag,)).fetchone()
    if row is None:
        return ToolResult(False, error_code="TAG_NOT_FOUND", data={"field": "baggage_tag"},
                          message=f"No baggage record exists for tag {tag}.")
    return ToolResult(True, data=dict(row))


@safe_tool
def find_baggage_by_booking(db_path, booking_reference) -> ToolResult:
    """List the baggage tags registered on a booking (used to suggest a corrected tag)."""
    ref = str(booking_reference).strip().upper()
    with get_conn(db_path) as conn:
        rows = conn.execute("SELECT baggage_tag FROM baggage WHERE booking_reference = ?", (ref,)).fetchall()
    return ToolResult(True, data={"booking_reference": ref, "tags": [r["baggage_tag"] for r in rows]})
