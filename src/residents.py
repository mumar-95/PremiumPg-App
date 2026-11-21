"""Resident rent lookup — self-service only; never expose other residents."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import DOCUMENTS_DIR

RESIDENTS_CSV = DOCUMENTS_DIR / "synthetic_residents_rent.csv"


def load_residents(path: Path | None = None) -> pd.DataFrame:
    """Load synthetic resident rent records (internal use / self-lookup only)."""
    path = path or RESIDENTS_CSV
    df = pd.read_csv(path)
    df["nickname"] = df["nickname"].astype(str).str.strip()
    df["room"] = df["room"].astype(str).str.strip()
    df["rent_inr"] = df["rent_inr"].astype(int)
    df["due_date"] = df["due_date"].astype(str).str.strip()
    df["paid_status"] = df["paid_status"].astype(str).str.strip()
    return df


def lookup_own_record(
    df: pd.DataFrame,
    nickname: str,
    room: str,
) -> dict | None:
    """
    Return a single resident's rent record only when nickname AND room match.

    Both fields are required so guessing a nickname alone cannot list others.
    Never returns other residents' rows.
    """
    nick = (nickname or "").strip()
    room_id = (room or "").strip()
    if not nick or not room_id:
        return None

    matches = df[
        (df["nickname"].str.casefold() == nick.casefold())
        & (df["room"].str.casefold() == room_id.casefold())
    ]
    if matches.empty:
        return None

    row = matches.iloc[0]
    return {
        "nickname": row["nickname"],
        "room": row["room"],
        "rent_inr": int(row["rent_inr"]),
        "due_date": row["due_date"],
        "paid_status": row["paid_status"],
    }


def known_room_ids(df: pd.DataFrame) -> list[str]:
    """Room IDs that appear in resident data (for selectbox UX only — not a directory)."""
    return sorted(df["room"].unique().tolist())
