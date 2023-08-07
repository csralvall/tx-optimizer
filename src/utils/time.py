SECONDS_IN_MIN = 60
SECONDS_IN_HOUR = SECONDS_IN_MIN * 60
SECONDS_IN_DAY = SECONDS_IN_HOUR * 24
SECONDS_IN_WEEK = SECONDS_IN_DAY * 7

TIME_DURATION_UNITS = {
    "week": SECONDS_IN_WEEK,
    "day": SECONDS_IN_WEEK,
    "hour": SECONDS_IN_HOUR,
    "min": SECONDS_IN_MIN,
    "sec": 1,
}


def human_readable_elapsed_time(seconds: float) -> str:
    if seconds == 0:
        return "inf"
    parts: list[str] = []
    remaining_seconds: float = seconds
    for unit, elapsed_seconds in TIME_DURATION_UNITS.items():
        amount, remaining_seconds = divmod(
            int(remaining_seconds), elapsed_seconds
        )
        cardinality: str = "" if amount == 1 else "s"
        if amount > 0:
            parts.append(f"{amount} {unit}{cardinality}")
    return ", ".join(parts)
