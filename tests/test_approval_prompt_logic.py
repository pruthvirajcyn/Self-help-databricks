from self_help_agent.schemas import ReviewDecision


def test_review_decision_schema():
    decision = ReviewDecision(approved=False, feedback="Make it easier")
    assert decision.approved is False
    assert decision.feedback == "Make it easier"
