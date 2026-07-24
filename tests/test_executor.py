from langchain_core.messages import HumanMessage

from self_help_agent.graph import executor_node
from self_help_agent.schemas import CoachState, DailyTask


def test_executor_creates_one_event_per_day():
    state = CoachState(
        messages=[HumanMessage(content="test")],
        skill="public speaking",
        plan_days=[
            [DailyTask(hour="0-10 min", activity="Record a short intro", goal="Build baseline")],
            [DailyTask(hour="0-10 min", activity="Practice vocal clarity", goal="Improve delivery")],
        ],
    )

    output = executor_node(state)
    assert len(output["calendar_events"]) == 2
    assert output["calendar_events"][0]["title"].startswith("[public speaking] Day 1")
