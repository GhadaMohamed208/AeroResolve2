"""Generate the synthetic AeroResolve datasets.

Creates (deterministically, seed=42):
    data/flights.csv          passenger bookings on flights
    data/baggage.csv          baggage tags, statuses and last known locations
    data/claims.csv           a few pre-existing claims (used to test duplicate detection)
    data/baggage_policy.pdf   the (fictional) baggage policy used for RAG

Everything is FICTIONAL. "AeroResolve" is a prototype and is not affiliated with any airline.

Run:  python scripts/generate_data.py
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

AIRPORTS = {
    "CAI": "Cairo Airport", "DXB": "Dubai International Airport", "LHR": "London Heathrow Airport",
    "FRA": "Frankfurt Airport", "IST": "Istanbul Airport", "JED": "Jeddah King Abdulaziz Airport",
    "AMM": "Amman Queen Alia Airport", "CDG": "Paris Charles de Gaulle Airport",
    "FCO": "Rome Fiumicino Airport", "HRG": "Hurghada Airport", "SSH": "Sharm El Sheikh Airport",
}

# Hand-crafted rows that drive the three demos and the edge-case tests.
# (booking, flight, date, origin, dest, passenger, tag, status, location, last_updated)
DEMO_ROWS = [
    # Demo A - Delayed
    ("BK1023", "FL123", "2026-09-20", "CAI", "DXB", "Ghada Mohamed", "BG77821", "Delayed",
     "Cairo Airport", "2026-09-20 18:30"),
    # Demo B - Delivered
    ("BK2045", "FL210", "2026-09-19", "CAI", "LHR", "Omar Hassan", "BG50412", "Delivered",
     "London Heathrow Airport - Carousel 7", "2026-09-19 21:15"),
    # Demo C - Not Found
    ("BK3078", "FL305", "2026-09-21", "CAI", "FRA", "Layla Ibrahim", "BG90233", "Not Found",
     "Cairo Airport - Sorting Hall B (last scan)", "2026-09-21 09:40"),
    # In Transit
    ("BK4012", "FL450", "2026-09-21", "CAI", "IST", "Youssef Khaled", "BG61150", "In Transit",
     "Istanbul Airport - Transfer Hub", "2026-09-21 14:05"),
    # At Airport
    ("BK5533", "FL512", "2026-09-22", "HRG", "CAI", "Nour Ali", "BG33871", "At Airport",
     "Cairo Airport - Baggage Service Office", "2026-09-22 08:20"),
    # Under Investigation, claim already on file (CLM-1002)
    ("BK6120", "FL123", "2026-09-20", "CAI", "DXB", "Karim Nabil", "BG18044", "Under Investigation",
     "Cairo Airport - Central Tracing", "2026-09-21 10:12"),
    # Delayed, claim already on file (CLM-1001) -> duplicate-claim scenario
    ("BK7071", "FL123", "2026-09-20", "CAI", "DXB", "Sara Mostafa", "BG45120", "Delayed",
     "Cairo Airport", "2026-09-20 19:05"),
    # Under Investigation, NO claim yet -> claim gets created
    ("BK6288", "FL640", "2026-09-18", "JED", "CAI", "Mahmoud Adel", "BG27750", "Under Investigation",
     "Jeddah King Abdulaziz Airport - Central Tracing", "2026-09-19 07:45"),
]

EXISTING_CLAIMS = [
    # claim_id, tag, booking, passenger, flight, date, type, status, location, email, created_at
    ("CLM-1001", "BG45120", "BK7071", "Sara Mostafa", "FL123", "2026-09-20", "Delayed Baggage", "Open",
     "Cairo Airport", "sara.mostafa@example.com", "2026-09-20 19:20:00"),
    ("CLM-1002", "BG18044", "BK6120", "Karim Nabil", "FL123", "2026-09-20", "Baggage Investigation",
     "Under Investigation", "Cairo Airport - Central Tracing", "karim.nabil@example.com",
     "2026-09-21 10:12:00"),
]

FIRST = ["Ahmed", "Mona", "Hassan", "Salma", "Tarek", "Dina", "Khaled", "Farida", "Amr", "Yasmin",
         "Ziad", "Heba", "Sherif", "Rana", "Mostafa", "Aya", "Ibrahim", "Reem", "Adam", "Lina",
         "James", "Emma", "Lucas", "Sofia", "Daniel", "Maria"]
LAST = ["Farouk", "Saleh", "Mansour", "Fathy", "Gamal", "Sabry", "Nasser", "Hegazy", "Zaki",
        "Abdallah", "Shawky", "Ramadan", "Taha", "Lotfy", "Samir", "Morsi", "Wagdy", "Kamel",
        "Smith", "Martin", "Garcia", "Rossi"]

STATUS_WEIGHTS = [("Delivered", 50), ("In Transit", 15), ("At Airport", 10), ("Delayed", 15),
                  ("Not Found", 5), ("Under Investigation", 5)]


def _location(status: str, origin: str, dest: str, rng: random.Random) -> str:
    o, d = AIRPORTS[origin], AIRPORTS[dest]
    return {
        "Delivered": f"{d} - Carousel {rng.randint(1, 9)}",
        "In Transit": f"{rng.choice([a for a in AIRPORTS.values() if a not in (o, d)])} - Transfer Hub",
        "At Airport": f"{d} - Baggage Service Office",
        "Delayed": f"{o}",
        "Not Found": f"{o} - Sorting Hall {rng.choice('ABC')} (last scan)",
        "Under Investigation": f"{o} - Central Tracing",
    }[status]


def build_rows() -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(42)
    flights, baggage = [], []
    used_bk = {r[0] for r in DEMO_ROWS}
    used_bg = {r[6] for r in DEMO_ROWS}

    def add(bk, fn, date, o, d, name, tag, status, loc, upd):
        flights.append(dict(flight_number=fn, flight_date=date, origin=o, destination=d,
                            booking_reference=bk, passenger_name=name))
        baggage.append(dict(baggage_tag=tag, booking_reference=bk, flight_number=fn, status=status,
                            last_known_location=loc, last_updated=upd))

    for row in DEMO_ROWS:
        bk, fn, date, o, d, name, tag, status, loc, upd = row
        add(bk, fn, date, o, d, name, tag, status, AIRPORTS.get(loc, loc), upd)

    statuses = [s for s, w in STATUS_WEIGHTS for _ in range(w)]
    codes = list(AIRPORTS)
    for _ in range(32):
        while True:
            bk = f"BK{rng.randint(1000, 9999)}"
            if bk not in used_bk:
                used_bk.add(bk)
                break
        while True:
            tag = f"BG{rng.randint(10000, 99999)}"
            if tag not in used_bg:
                used_bg.add(tag)
                break
        o, d = rng.sample(codes, 2)
        day = rng.randint(15, 22)
        fn = f"FL{rng.randint(100, 899)}"
        status = rng.choice(statuses)
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        upd = f"2026-09-{min(day + rng.randint(0, 1), 22):02d} {rng.randint(6, 22):02d}:{rng.choice(['00', '15', '30', '45'])}"
        add(bk, fn, f"2026-09-{day:02d}", o, d, name, tag, status, _location(status, o, d, rng), upd)

    claims = [dict(claim_id=c[0], baggage_tag=c[1], booking_reference=c[2], passenger_name=c[3],
                   flight_number=c[4], flight_date=c[5], claim_type=c[6], status=c[7],
                   last_known_location=c[8], contact_email=c[9], created_at=c[10])
              for c in EXISTING_CLAIMS]
    return flights, baggage, claims


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


POLICY = [
    ("1. Purpose and Scope",
     ["This document describes the baggage handling and claim rules of the fictional demonstration "
      "airline used by the AeroResolve prototype. It is synthetic content created for a university "
      "project and is not the policy of any real airline."]),
    ("2. Delayed Baggage",
     ["Baggage is considered delayed when it does not arrive on the same flight as the passenger. "
      "Once a delayed status is confirmed in the tracing system, a baggage claim is opened "
      "automatically. Most delayed bags are located and delivered within 24 to 48 hours. Recovered "
      "baggage is delivered free of charge to the address given by the passenger."]),
    ("3. Lost Baggage and Investigations",
     ["When baggage cannot be located by the tracing system, its status becomes Not Found and a "
      "lost baggage investigation is opened. The airline searches for the baggage for 21 days. "
      "If the baggage has not been found after 21 days it is declared lost and compensation "
      "can be requested. Investigations are handled by the Central Tracing team."]),
    ("4. Filing a Baggage Claim",
     ["To file a claim the passenger must provide the passenger name, booking reference, flight "
      "number, flight date, baggage tag number and a contact email address. Delayed or missing "
      "baggage must be reported as soon as possible and no later than 21 days after the flight "
      "arrival date. Only one active claim can exist for each baggage tag."]),
    ("5. Interim Expenses",
     ["While baggage is delayed, passengers may buy reasonable essential items such as toiletries "
      "and basic clothing. These expenses are reimbursed up to USD 75 per day for a maximum of "
      "5 days. Original receipts must be submitted with the claim."]),
    ("6. Compensation Limits",
     ["Compensation for baggage declared lost is limited to USD 1,500 per passenger and is subject "
      "to proof of the value of the contents. Compensation is paid within 30 days after the claim "
      "is approved."]),
    ("7. Baggage Status Definitions",
     ["Delivered means the baggage was handed to the passenger. Delayed means it did not travel "
      "with the passenger and is being forwarded. In Transit means it is moving between airports. "
      "At Airport means it has been located and is waiting for collection or delivery. Not Found "
      "means the tracing system has no current location. Under Investigation means the Central "
      "Tracing team is actively searching."]),
    ("8. Baggage That Was Already Delivered",
     ["No claim is created for baggage whose status is Delivered. If a passenger believes the "
      "delivered baggage is damaged or items are missing, the passenger must report this at the "
      "airport baggage service office or contact customer support with photographs."]),
    ("9. Follow-up and Communication",
     ["Passengers with an open claim receive an email confirmation with the claim number. Status "
      "updates are sent at least every 48 hours until the claim is closed. Baggage that is In "
      "Transit or At Airport is monitored and the passenger is informed of progress without "
      "opening a claim."]),
    ("10. Escalation and Manual Review",
     ["If the baggage tag cannot be matched to any record, the case is escalated to a human agent "
      "for manual review. A human agent contacts the passenger within one business day."]),
]


def build_policy_pdf(path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=15)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm, title="Baggage Policy (Demo)",
                            author="AeroResolve prototype")
    story = [Paragraph("Baggage Handling and Claims Policy", styles["Title"]),
             Paragraph("Fictional demonstration airline - AeroResolve prototype", styles["Italic"]),
             Spacer(1, 14)]
    for heading, paragraphs in POLICY:
        story.append(Paragraph(heading, styles["Heading2"]))
        for text in paragraphs:
            story.append(Paragraph(text, body))
        story.append(Spacer(1, 6))
    doc.build(story)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    flights, baggage, claims = build_rows()
    write_csv(DATA / "flights.csv", flights)
    write_csv(DATA / "baggage.csv", baggage)
    write_csv(DATA / "claims.csv", claims)
    build_policy_pdf(DATA / "baggage_policy.pdf")
    print(f"flights: {len(flights)}  baggage: {len(baggage)}  claims: {len(claims)}  + policy pdf")


if __name__ == "__main__":
    main()
