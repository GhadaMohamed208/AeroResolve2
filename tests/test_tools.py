"""Unit tests for the tool layer (Requirement 1: Data Retrieval / Requirement 3: Automated Actions).

These call the tools directly, bypassing the LLM/agents entirely, to prove that
retrieval and persistence are correct regardless of what any model does.
"""
from __future__ import annotations

from app.tools.baggage_tools import check_booking, find_baggage_by_booking, get_flight_information, track_baggage
from app.tools.claim_tools import create_baggage_claim, create_follow_up, get_existing_claim, save_email
from app.tools.data_tools import export_claims_csv, search_policy


# ------------------------------------------------------------- baggage_tools
def test_check_booking_success(services):
    r = check_booking(services.db_path, "BK1023", passenger_name="Ghada Mohamed",
                       flight_number="FL123", flight_date="2026-09-20")
    assert r.ok
    assert r.data["booking_reference"] == "BK1023"


def test_check_booking_unknown_reference(services):
    r = check_booking(services.db_path, "BK0000")
    assert not r.ok
    assert r.error_code == "BOOKING_NOT_FOUND"


def test_check_booking_name_mismatch(services):
    r = check_booking(services.db_path, "BK1023", passenger_name="Someone Else")
    assert not r.ok
    assert r.error_code == "NAME_MISMATCH"


def test_track_baggage_found(services):
    r = track_baggage(services.db_path, "BG77821")
    assert r.ok
    assert r.data["status"] == "Delayed"
    assert r.data["last_known_location"] == "Cairo Airport"


def test_track_baggage_not_found_tag(services):
    r = track_baggage(services.db_path, "BG00000")
    assert not r.ok
    assert r.error_code == "TAG_NOT_FOUND"


def test_get_flight_information(services):
    r = get_flight_information(services.db_path, "FL123", "2026-09-20")
    assert r.ok
    assert r.data["origin"] == "CAI" and r.data["destination"] == "DXB"


def test_find_baggage_by_booking_suggests_tag(services):
    r = find_baggage_by_booking(services.db_path, "BK1023")
    assert r.ok
    assert "BG77821" in r.data["tags"]


# --------------------------------------------------------------- claim_tools
def test_create_baggage_claim_for_delayed(services):
    r = create_baggage_claim(services.db_path, baggage_tag="BG77821", booking_reference="BK1023",
                              passenger_name="Ghada Mohamed", flight_number="FL123", flight_date="2026-09-20",
                              claim_type="Delayed Baggage", contact_email="ghada@example.com")
    assert r.ok
    assert r.data["created"] is True
    assert r.data["claim"]["claim_id"].startswith("CLM-")
    assert r.data["claim"]["status"] == "Open"
    assert r.data["claim"]["last_known_location"] == "Cairo Airport"  # taken from DB, not from the caller


def test_create_baggage_claim_refuses_delivered(services):
    r = create_baggage_claim(services.db_path, baggage_tag="BG50412", booking_reference="BK2045",
                              passenger_name="Omar Hassan", flight_number="FL210", flight_date="2026-09-19",
                              claim_type="Delayed Baggage", contact_email="omar@example.com")
    assert not r.ok
    assert r.error_code == "BAGGAGE_ALREADY_DELIVERED"


def test_create_baggage_claim_is_idempotent_no_duplicate(services):
    """Two calls for the same delayed bag must yield exactly one active claim."""
    first = create_baggage_claim(services.db_path, baggage_tag="BG77821", booking_reference="BK1023",
                                  passenger_name="Ghada Mohamed", flight_number="FL123", flight_date="2026-09-20",
                                  claim_type="Delayed Baggage", contact_email="ghada@example.com")
    second = create_baggage_claim(services.db_path, baggage_tag="BG77821", booking_reference="BK1023",
                                   passenger_name="Ghada Mohamed", flight_number="FL123", flight_date="2026-09-20",
                                   claim_type="Delayed Baggage", contact_email="ghada@example.com")
    assert first.ok and second.ok
    assert first.data["created"] is True
    assert second.data["created"] is False
    assert second.data["duplicate"] is True
    assert first.data["claim"]["claim_id"] == second.data["claim"]["claim_id"]


def test_get_existing_claim_pre_seeded_duplicate(services):
    """BG45120 ships with an existing open claim (CLM-1001) in the fixtures."""
    r = get_existing_claim(services.db_path, "BG45120")
    assert r.ok
    assert r.data["claim"] is not None
    assert r.data["claim"]["claim_id"] == "CLM-1001"


def test_create_follow_up_persists(services):
    r = create_follow_up(services.db_path, baggage_tag="BG61150", booking_reference="BK4012",
                          follow_up_type="Monitor Baggage", note="test note", due_hours=12)
    assert r.ok
    assert r.data["follow_up_type"] == "Monitor Baggage"
    assert r.data["status"] == "Scheduled"


def test_save_email_persists(services):
    r = save_email(services.db_path, claim_id="CLM-1001", baggage_tag="BG45120",
                    recipient="sara@example.com", subject="Test", body="Body text")
    assert r.ok
    assert r.data["status"] == "Preview (not sent)"


# ---------------------------------------------------------------- data_tools
def test_search_policy_returns_relevant_section(services):
    r = search_policy(services.policy_path, "lost baggage investigation 21 days compensation", k=2)
    assert r.ok
    assert any("lost" in h["title"].lower() or "investigation" in h["title"].lower() for h in r.data["hits"])


def test_search_policy_missing_file_fails_gracefully(tmp_path):
    r = search_policy(str(tmp_path / "no_such_file.pdf"), "delayed baggage")
    assert not r.ok
    assert r.error_code == "POLICY_UNAVAILABLE"


def test_export_claims_csv(services):
    create_baggage_claim(services.db_path, baggage_tag="BG77821", booking_reference="BK1023",
                          passenger_name="Ghada Mohamed", flight_number="FL123", flight_date="2026-09-20",
                          claim_type="Delayed Baggage", contact_email="ghada@example.com")
    r = export_claims_csv(services.db_path, services.exports_dir)
    assert r.ok
    assert r.data["rows"] >= 1
