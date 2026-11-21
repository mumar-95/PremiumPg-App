"""Load rooms CSV and policy DOCX into structured data."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from docx import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
ROOMS_CSV = DOCUMENTS_DIR / "pg_rooms.csv"
POLICY_DOCX = DOCUMENTS_DIR / "Mens_PG_Resident_Policy.docx"

def load_rooms() -> pd.DataFrame:
    """Load room inventory and normalize types."""
    df = pd.read_csv(ROOMS_CSV)
    df["sharing"] = df["sharing"].astype(int)
    df["rent_inr_per_bed_month"] = df["rent_inr_per_bed_month"].astype(int)
    df["beds_free"] = df["beds_free"].astype(int)
    df["AC"] = df["AC"].astype(str).str.strip()
    df["food_plan"] = df["food_plan"].astype(str).str.strip()
    df["amenities"] = df["amenities"].astype(str).str.strip()
    return df


def _paragraphs_from_docx(path: Path) -> list[str]:
    doc = Document(str(path))
    return [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]


def load_policy_sections(path: Path | None = None) -> list[dict[str, str]]:
    """
    Split the resident policy into citation-friendly sections.

    Returns list of dicts: {section_id, section_title, content}
    """
    path = path or POLICY_DOCX
    paragraphs = _paragraphs_from_docx(path)

    sections: list[dict[str, str]] = []
    current_id = "intro"
    current_title = "Introduction"
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if body:
            sections.append(
                {
                    "section_id": current_id,
                    "section_title": current_title,
                    "content": body,
                }
            )
        current_lines = []

    for line in paragraphs:
        # Skip template header blanks / title chrome
        if line.startswith("_") or line in {
            "MEN'S PG RESIDENT POLICY & HOUSE RULES",
            "Resident Handbook | Sample Template",
            "PG / Property Name",
            "Property Address",
            "Policy Effective Date",
            "Manager / Emergency Contact",
            "Resident Acknowledgement",
            "Resident Full Name",
            "Room / Bed Number",
            "Resident Signature & Date",
            "Manager / Authorised Representative",
        }:
            continue

        purpose_match = re.match(r"^Purpose\.?\s+(.*)$", line, re.IGNORECASE)
        numbered = re.match(r"^(\d+)\.\s+(.+)$", line)

        if purpose_match:
            flush()
            current_id = "Purpose"
            current_title = "Purpose"
            rest = purpose_match.group(1).strip()
            current_lines = [rest] if rest else []
            continue

        if numbered:
            flush()
            num, title = numbered.group(1), numbered.group(2).strip()
            current_id = f"Section {num}"
            current_title = title
            current_lines = []
            continue

        # Stop collecting after acknowledgement boilerplate starts repeating
        if line.startswith("I confirm that I have received"):
            flush()
            current_id = "Resident Acknowledgement"
            current_title = "Resident Acknowledgement"
            current_lines = [line]
            continue

        current_lines.append(line)

    flush()

    # Keep policy body sections useful for RAG (Purpose + 1–10 + acknowledgement summary)
    useful = [
        s
        for s in sections
        if s["section_id"] == "Purpose"
        or s["section_id"].startswith("Section ")
        or s["section_id"] == "Resident Acknowledgement"
    ]
    return useful if useful else sections


def list_amenity_options(df: pd.DataFrame) -> list[str]:
    """Unique amenity tokens across rooms."""
    tokens: set[str] = set()
    for raw in df["amenities"].dropna():
        for part in str(raw).split(";"):
            token = part.strip()
            if token:
                tokens.add(token)
    return sorted(tokens)
