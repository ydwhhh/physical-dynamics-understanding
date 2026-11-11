"""Closed-set labels and prediction parsing."""

from __future__ import annotations

import re

LABELS = ("Dropping Ball", "Pendulum", "Sliding Cone", "Hitting Cones")
CHOICE_TO_LABEL = dict(zip(("A", "B", "C", "D"), LABELS))


def normalize_label(value: str) -> str:
    """Return the canonical class name or raise for an unsupported label."""
    cleaned = " ".join(str(value).strip().split()).casefold()
    matches = {label.casefold(): label for label in LABELS}
    if cleaned not in matches:
        raise ValueError(f"unknown label {value!r}; expected one of {list(LABELS)}")
    return matches[cleaned]


def parse_choice(response: str) -> tuple[str | None, str | None]:
    """Parse a strict single A/B/C/D response.

    Surrounding whitespace and one trailing punctuation character are accepted.
    Explanations and ambiguous answers are deliberately invalid.
    """
    match = re.fullmatch(r"\s*([ABCD])\s*[.)]?\s*", response, flags=re.IGNORECASE)
    if not match:
        return None, None
    choice = match.group(1).upper()
    return choice, CHOICE_TO_LABEL[choice]
