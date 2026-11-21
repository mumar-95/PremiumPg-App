"""Session-only booking request helpers."""

from __future__ import annotations

import secrets
from datetime import date
from typing import Any

from src.privacy import mask_phone


def new_request_id() -> str:
    return f"REQ-{secrets.token_hex(4).upper()}"


def create_booking(
    *,
    room_id: str,
    move_in_date: date,
    name: str,
    contact: str,
) -> dict[str, Any]:
    return {
        "request_id": new_request_id(),
        "room_id": room_id,
        "move_in_date": move_in_date.isoformat(),
        "name": (name or "").strip(),
        "contact": (contact or "").strip(),
        "contact_masked": mask_phone((contact or "").strip()),
        "status": "Pending",
    }


def bookings_for_display(bookings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return copies with contact masked for UI tables."""
    rows = []
    for b in bookings:
        rows.append(
            {
                "request_id": b["request_id"],
                "room_id": b["room_id"],
                "move_in_date": b["move_in_date"],
                "name": b.get("name", ""),
                "contact": b.get("contact_masked") or mask_phone(b.get("contact", "")),
                "status": b.get("status", "Pending"),
            }
        )
    return rows
