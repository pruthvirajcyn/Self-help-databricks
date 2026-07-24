import pytest
from pydantic import ValidationError

from self_help_agent.schemas import EnhancedIntent


def test_enhanced_intent_strips_skill():
    x = EnhancedIntent(
        skill="  public speaking  ",
        daily_minutes=40,
        duration_days=56,
        clarified_prompt="Improve public speaking.",
    )
    assert x.skill == "public speaking"


def test_enhanced_intent_rejects_tiny_time_budget():
    with pytest.raises(ValidationError):
        EnhancedIntent(
            skill="public speaking",
            daily_minutes=2,
            duration_days=30,
            clarified_prompt="Improve public speaking.",
        )
