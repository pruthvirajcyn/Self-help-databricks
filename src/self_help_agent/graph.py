from __future__ import annotations

import argparse
import re
import uuid
from typing import Any

from dotenv import load_dotenv

load_dotenv()

try:
    import mlflow
except Exception:  # pragma: no cover - keeps local import flexible
    mlflow = None

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from self_help_agent.config import DEFAULT_CONFIG
from self_help_agent.prompts import (
    APPROVAL_SYSTEM,
    ENHANCER_SYSTEM,
    PLANNER_SYSTEM,
    RESEARCHER_SYSTEM,
    REVIEWER_SYSTEM,
    SUPERVISOR_SYSTEM,
)
from self_help_agent.schemas import (
    CoachState,
    EnhancedIntent,
    PlanDraft,
    ResearchFindings,
    ReviewDecision,
    SupervisorDecision,
)

llm = ChatOpenAI(
    model=DEFAULT_CONFIG.llm_model,
    temperature=DEFAULT_CONFIG.temperature,
)

tavily = TavilySearch(max_results=DEFAULT_CONFIG.tavily_max_results)


def structured_output(schema: type):
    """Use function-calling mode while prototyping to avoid strict schema failures."""
    return llm.with_structured_output(schema, method="function_calling")


def _trace(name: str):
    """Optional MLflow trace decorator that works even when MLflow is unavailable."""
    if mlflow is not None and hasattr(mlflow, "trace"):
        return mlflow.trace(name=name)

    def decorator(fn):
        return fn

    return decorator


@_trace("supervisor_node")
def supervisor_node(state: CoachState) -> Command:
    """LLM-routed supervisor with explicit workflow-state context."""

    workflow_state = SystemMessage(
        content=f"""
Current workflow state:
skill: {state.skill}
daily_minutes: {state.daily_minutes}
duration_days: {state.duration_days}
has_resources: {state.resources is not None}
has_plan: {state.plan is not None}
approved: {state.approved}
review_feedback: {state.review_feedback}
paused: {state.paused}

Use this workflow state plus conversation history to choose the next node.
"""
    )

    messages = [SystemMessage(content=SUPERVISOR_SYSTEM), workflow_state] + list(state.messages)
    decision = structured_output(SupervisorDecision).invoke(messages)

    allowed = {"enhancer", "researcher", "planner", "reviewer", "executor"}
    goto = decision.next if decision.next in allowed else "planner"

    return Command(
        update={"messages": [AIMessage(content=f"[Supervisor] {decision.reason}", name="supervisor")]},
        goto=goto,
    )


@_trace("enhancer_node")
def enhancer_node(state: CoachState) -> dict:
    messages = [SystemMessage(content=ENHANCER_SYSTEM)] + list(state.messages)
    result = structured_output(EnhancedIntent).invoke(messages)

    summary = (
        f"[Enhancer] skill='{result.skill}', daily_minutes={result.daily_minutes}, "
        f"duration_days={result.duration_days}\nClarified: {result.clarified_prompt}"
    )

    return {
        "skill": result.skill,
        "daily_minutes": result.daily_minutes,
        "duration_days": result.duration_days,
        "messages": [AIMessage(content=summary, name="enhancer")],
    }


@_trace("researcher_node")
def researcher_node(state: CoachState) -> dict:
    skill = state.skill or "self-improvement"
    daily = state.daily_minutes or 30
    duration = state.duration_days or 30

    query = f"evidence-based beginner exercises and resources for {skill}"
    tavily_results = tavily.invoke({"query": query})

    results_list: list[dict[str, Any]] = []
    if isinstance(tavily_results, dict) and "results" in tavily_results:
        results_list = tavily_results["results"]

    context_lines: list[str] = []
    for i, result in enumerate(results_list[:5], start=1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        snippet = result.get("content") or result.get("snippet", "")
        snippet = (snippet[:300] + "…") if len(snippet) > 300 else snippet
        context_lines.append(f"{i}. {title}\n{url}\n{snippet}")

    researcher_user = f"""
Skill: {skill}
Daily minutes: {daily}
Duration days: {duration}

External findings:
{chr(10).join(context_lines)}

Combine external findings with your own knowledge.
Return only the most useful 3-4 resources.
"""

    findings = structured_output(ResearchFindings).invoke(
        [SystemMessage(content=RESEARCHER_SYSTEM), HumanMessage(content=researcher_user)]
    )

    rendered = "\n".join(f"- {r.title} — {r.url} — why: {r.why}" for r in findings.resources)
    return {
        "resources": findings.resources,
        "messages": [AIMessage(content=f"[Researcher]\n{rendered}", name="researcher")],
    }


@_trace("planner_node")
def planner_node(state: CoachState) -> dict:
    skill = state.skill or "self-improvement"
    daily = state.daily_minutes or 30
    duration = state.duration_days or 30
    resources = state.resources or []

    resource_bullets = "\n".join(f"- {r.title} ({r.url}) — {r.why}" for r in resources)
    revision_text = ""
    if state.approved is False and state.review_feedback:
        revision_text = f"\nUser requested revisions:\n{state.review_feedback}\n"

    prompt = (
        PLANNER_SYSTEM.format(daily_minutes=daily, duration_days=duration, skill=skill)
        + f"\nResources:\n{resource_bullets}\n"
        + revision_text
    )

    plan_draft = structured_output(PlanDraft).invoke([SystemMessage(content=prompt)])

    plan_text = f"Summary: {plan_draft.summary}\n\n"
    for day_num, tasks in enumerate(plan_draft.days, start=1):
        plan_text += f"Day {day_num}:\n"
        for task in tasks:
            plan_text += f"  - {task.hour}: {task.activity} (Goal: {task.goal})\n"
        plan_text += "\n"

    return {
        "plan": plan_text,
        "plan_days": plan_draft.days,
        "approved": None,
        "review_feedback": None,
        "paused": False,
        "messages": [AIMessage(content=f"[Planner]\n{plan_text}", name="planner")],
    }


@_trace("reviewer_node")
def reviewer_node(state: CoachState) -> dict:
    plan_text = state.plan or "No plan available."
    reviewer_prompt = f"""
Review this training plan for clarity, realism, safety, and effectiveness:

{plan_text}

Give a brief critique with strengths, weaknesses, and what the user should verify.
Do NOT decide approval.
"""

    critique = llm.invoke(
        [SystemMessage(content=REVIEWER_SYSTEM), HumanMessage(content=reviewer_prompt)]
    )

    return {
        "messages": [
            AIMessage(content=f"[Reviewer] {critique.content}", name="reviewer"),
            AIMessage(
                content="[Reviewer] Do you approve this plan? Reply 'yes', or describe what needs to change.",
                name="reviewer",
            ),
        ],
        "review_feedback": critique.content,
        "paused": True,
    }


@_trace("approval_node")
def approval_node(state: CoachState) -> dict:
    latest_user_message = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            latest_user_message = str(msg.content)
            break

    approval_prompt = f"""
The user was asked to approve a self-help coaching plan.

User response:
{latest_user_message}

Decide whether the user approved the plan.
"""

    decision = structured_output(ReviewDecision).invoke(
        [SystemMessage(content=APPROVAL_SYSTEM), HumanMessage(content=approval_prompt)]
    )

    if decision.approved:
        msg = AIMessage(content="[Approval] User approved the plan.", name="approval")
    else:
        msg = AIMessage(
            content=f"[Approval] User requested changes: {decision.feedback}",
            name="approval",
        )

    return {
        "approved": decision.approved,
        "review_feedback": decision.feedback,
        "paused": False,
        "messages": [msg],
    }


@_trace("executor_node")
def executor_node(state: CoachState) -> dict:
    skill = state.skill or "Skill"
    default_time = "19:00"
    plan_days = state.plan_days or []

    events: list[dict[str, Any]] = []
    notifications: list[str] = []

    for day_num, tasks in enumerate(plan_days, start=1):
        main_focus = tasks[0].activity.strip().rstrip(".") if tasks else "Focus session"
        title = f"[{skill}] Day {day_num}: {main_focus[:60]}"
        description = "\n".join(
            f"- {task.hour}: {task.activity} (Goal: {task.goal})" for task in tasks
        )
        events.append(
            {
                "day": day_num,
                "start_local_time": default_time,
                "title": title,
                "description": description,
            }
        )
        notifications.append(f"Reminder: {title} at {default_time} today. You got this.")

    preview = "\n".join(
        f"- Day {event['day']} @ {event['start_local_time']}: {event['title']}"
        for event in events[:10]
    )
    if len(events) > 10:
        preview += f"\n...and {len(events) - 10} more days scheduled."

    return {
        "calendar_events": events,
        "notifications": notifications,
        "messages": [
            AIMessage(content="[Executor] Created calendar-style events and reminders.", name="executor"),
            AIMessage(content=f"[Executor Preview]\n{preview}", name="executor"),
        ],
    }


def build_graph(interrupt_after_reviewer: bool = False):
    graph = StateGraph(CoachState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("enhancer", enhancer_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("planner", planner_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("approval", approval_node)
    graph.add_node("executor", executor_node)

    graph.add_edge(START, "supervisor")
    graph.add_edge("enhancer", "supervisor")
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("planner", "supervisor")
    graph.add_edge("reviewer", "approval")
    graph.add_edge("approval", "supervisor")
    graph.add_edge("executor", END)

    checkpointer = InMemorySaver() if interrupt_after_reviewer else None
    kwargs = {}
    if interrupt_after_reviewer:
        kwargs["checkpointer"] = checkpointer
        kwargs["interrupt_after"] = ["reviewer"]
    return graph.compile(**kwargs)


def run_until_review(user_text: str) -> CoachState:
    app = build_graph(interrupt_after_reviewer=True)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial = CoachState(messages=[HumanMessage(content=user_text, name="user")])
    result = app.invoke(initial, config=config)
    return CoachState.model_validate(result)


def run_with_approval(user_text: str, approval_text: str = "yes") -> CoachState:
    app = build_graph(interrupt_after_reviewer=True)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial = CoachState(messages=[HumanMessage(content=user_text, name="user")])
    app.invoke(initial, config=config)
    app.update_state(
        config,
        {"messages": [HumanMessage(content=approval_text, name="user")], "paused": False},
    )
    result = app.invoke(None, config=config)
    return CoachState.model_validate(result)


def state_to_dict(state: CoachState) -> dict[str, Any]:
    return {
        "skill": state.skill,
        "daily_minutes": state.daily_minutes,
        "duration_days": state.duration_days,
        "resources": [r.model_dump() for r in state.resources or []],
        "plan": state.plan,
        "plan_days": [
            [task.model_dump() for task in day] for day in state.plan_days or []
        ],
        "approved": state.approved,
        "review_feedback": state.review_feedback,
        "calendar_events": state.calendar_events or [],
        "notifications": state.notifications or [],
        "final_response": state.plan or "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("user_text")
    parser.add_argument("--approve", default="yes")
    args = parser.parse_args()
    final_state = run_with_approval(args.user_text, args.approve)
    print(state_to_dict(final_state))
