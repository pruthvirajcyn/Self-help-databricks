from __future__ import annotations

try:
    from mlflow.genai.scorers import scorer
except Exception:  # pragma: no cover
    def scorer(fn=None, **kwargs):
        def wrap(f):
            return f
        return wrap(fn) if fn else wrap


@scorer
def plan_has_daily_tasks(inputs, outputs, expectations=None):
    """Checks whether the generated plan contains the expected day structure."""
    plan = outputs.get("plan", "") if isinstance(outputs, dict) else ""
    if not plan:
        return 0.0

    expected_days = None
    if isinstance(inputs, dict):
        expected_days = inputs.get("expected_duration_days")
    if expectations and isinstance(expectations, dict):
        expected_days = expected_days or expectations.get("expected_duration_days")

    if not expected_days:
        return 1.0 if "Day 1" in plan else 0.0

    expected_days = int(expected_days)
    hits = sum(1 for day in range(1, expected_days + 1) if f"Day {day}" in plan)
    return hits / float(expected_days)


@scorer
def executor_event_count_matches_duration(inputs, outputs, expectations=None):
    """Checks whether executor generated one event per planned day."""
    if not isinstance(outputs, dict):
        return 0.0

    expected_days = None
    if isinstance(inputs, dict):
        expected_days = inputs.get("expected_duration_days")
    if expectations and isinstance(expectations, dict):
        expected_days = expected_days or expectations.get("expected_duration_days")

    events = outputs.get("calendar_events", [])
    if not expected_days:
        return 1.0 if events else 0.0

    return 1.0 if len(events) == int(expected_days) else 0.0


@scorer
def approval_classification_correct(inputs, outputs, expectations=None):
    expected = None
    if isinstance(inputs, dict):
        expected = inputs.get("expected_approved")
    if expectations and isinstance(expectations, dict):
        expected = expected if expected is not None else expectations.get("expected_approved")
    if expected is None:
        return 1.0
    actual = outputs.get("approved") if isinstance(outputs, dict) else None
    return 1.0 if actual == expected else 0.0
