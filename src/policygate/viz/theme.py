"""Colour and chrome tokens for the dashboard's charts, in both modes.

Values are the dataviz reference palette. The categorical pair was run through
its validator in light and dark (CVD delta-E 24.7 / 26.8, all checks pass).

Colour follows the entity, never its position: ``full`` is always slot 1 and
``baseline`` always slot 2, on every chart, so a reader who learns "full is
blue" is never misled by a later page. Every other arm is context and wears the
muted ink.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Mode = Literal["light", "dark"]


class Tokens(BaseModel):
    model_config = ConfigDict(frozen=True)

    surface: str
    text_primary: str
    text_secondary: str
    muted: str
    grid: str
    axis: str
    full: str
    baseline: str
    sequential: tuple[str, ...]
    """Light to dark, for magnitude."""

    font: str = 'system-ui, -apple-system, "Segoe UI", sans-serif'

    def arm_color(self, arm: str) -> str:
        if arm == "full":
            return self.full
        if arm == "baseline":
            return self.baseline
        return self.muted


_SEQUENTIAL = (
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
)

LIGHT = Tokens(
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    full="#2a78d6",
    baseline="#eb6834",
    sequential=_SEQUENTIAL,
)

DARK = Tokens(
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    full="#3987e5",
    baseline="#d95926",
    sequential=_SEQUENTIAL,
)


def tokens(mode: str | None) -> Tokens:
    """Tokens for a Streamlit theme type. Unknown or not-yet-reported means light."""
    return DARK if mode == "dark" else LIGHT


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def ink_on(hex_color: str, t: Tokens) -> str:
    """Primary ink or white for a label set inside a filled cell, by luminance."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return "#0b0b0b" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "#ffffff"
