"""Session-only complaint ticket helpers."""

from __future__ import annotations

import secrets
from typing import Any

CATEGORIES = ["Maintenance", "Noise", "Cleanliness", "Safety", "Other"]
PRIORITIES = ["Low", "Medium", "High"]
STATUSES = ["Open", "In Progress", "Closed"]


def new_ticket_id() -> str:
    return f"TKT-{secrets.token_hex(4).upper()}"


def create_complaint(
    *,
    category: str,
    priority: str,
    description: str,
) -> dict[str, Any]:
    return {
        "ticket_id": new_ticket_id(),
        "category": category,
        "priority": priority,
        "description": (description or "").strip(),
        "status": "Open",
    }


def update_status(ticket: dict[str, Any], status: str) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"Invalid status: {status}")
    ticket = dict(ticket)
    ticket["status"] = status
    return ticket
