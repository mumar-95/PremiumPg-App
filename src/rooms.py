"""Room filter helpers — only rooms with free beds."""

from __future__ import annotations

import pandas as pd


def free_beds_only(df: pd.DataFrame) -> pd.DataFrame:
    """Hard rule: show only rooms with at least one free bed."""
    return df[df["beds_free"] > 0].copy()


def filter_rooms(
    df: pd.DataFrame,
    *,
    sharing: int | None = None,
    ac: str | None = None,
    food_plan: str | None = None,
    amenities: list[str] | None = None,
    max_budget: int | None = None,
) -> pd.DataFrame:
    """
    Filter inventory. Always restricts to free beds first.

    ac: "Yes" | "No" | None (any)
    amenities: all selected tokens must appear in the room's amenities string
    """
    out = free_beds_only(df)

    if sharing is not None:
        out = out[out["sharing"] == sharing]

    if ac in ("Yes", "No"):
        out = out[out["AC"] == ac]

    if food_plan:
        out = out[out["food_plan"] == food_plan]

    if max_budget is not None:
        out = out[out["rent_inr_per_bed_month"] <= max_budget]

    if amenities:
        for token in amenities:
            token_l = token.strip().lower()
            if not token_l:
                continue
            out = out[
                out["amenities"].str.lower().str.contains(token_l, regex=False, na=False)
            ]

    return out.reset_index(drop=True)


def room_ids_with_free_beds(df: pd.DataFrame) -> list[str]:
    return free_beds_only(df)["room_id"].tolist()
