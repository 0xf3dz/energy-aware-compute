"""Prompt templates for the daily maritime briefing.

The prompt states the values, the computed trends, and the absent fields. It
forbids the model from inventing any number.
"""

from datetime import datetime

SYSTEM_PROMPT = (
    "You are the weather briefing assistant of a solar powered sailing vessel. "
    "You write a short daily maritime briefing for the crew.\n"
    "Rules:\n"
    "1. Use only the values in the supplied forecast table and computed trends. "
    "Never invent, round away, or extrapolate a measurement.\n"
    "2. State the time of every value you quote, in UTC, using the time format of the table.\n"
    "3. If a field is absent or marked n/a, say that the data is missing. Do not guess it.\n"
    "4. Report the safety warnings that are supplied. Do not create new warnings.\n"
    "5. Write in short factual sentences. No introduction, no closing summary.\n"
    "6. Answer with at most 220 words."
)

BRIEFING_TEMPLATE = """Forecast position: {latitude:.3f}, {longitude:.3f}
Briefing generated (UTC): {generated_at}
Forecast source status: {availability}{reason_line}

Hourly forecast:
{table}

Computed trends:
{trends}

Deterministic safety warnings:
{warnings}

Missing fields:
{missing}

Write the briefing: wind and sea state for the next 24 hours, the trend, the
best and worst window for work on deck, and the safety warnings."""


def build_prompt(
    *,
    compact_table: str,
    trends: dict,
    warnings: list[str],
    missing: list[str],
    latitude: float,
    longitude: float,
    generated_at: datetime,
    availability: str,
    reason: str | None = None,
) -> str:
    return BRIEFING_TEMPLATE.format(
        latitude=latitude,
        longitude=longitude,
        generated_at=generated_at.isoformat(),
        availability=availability,
        reason_line=f" ({reason})" if reason else "",
        table=compact_table,
        trends=_format_trends(trends),
        warnings="\n".join(f"- {note}" for note in warnings) if warnings else "- none",
        missing=", ".join(missing) if missing else "none",
    )


def _format_trends(trends: dict) -> str:
    lines = []
    for name, value in trends.items():
        if value is None:
            lines.append(f"- {name}: data unavailable")
        elif isinstance(value, dict):
            parts = ", ".join(f"{key} {item}" for key, item in value.items())
            lines.append(f"- {name}: {parts}")
        elif isinstance(value, list):
            lines.append(f"- {name}: {', '.join(value) if value else 'none'}")
        else:
            lines.append(f"- {name}: {value}")
    return "\n".join(lines)
