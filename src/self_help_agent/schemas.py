from __future__ import annotations

from typing import Annotated, List, Literal, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator


class Resource(BaseModel):
    title: str = Field(..., description="Title of the resource.")
    url: str = Field(..., description="Resource URL; empty string if unavailable.")
    why: str = Field(..., description="Why this resource matters.")


class DailyTask(BaseModel):
    hour: str = Field(..., description="Time block, e.g. '0-10 min'.")
    activity: str = Field(..., description="Specific task to perform.")
    goal: str = Field(..., description="Measurable outcome for this task.")


class CoachState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)

    skill: Optional[str] = None
    daily_minutes: Optional[int] = None
    duration_days: Optional[int] = None

    resources: Optional[List[Resource]] = None

    plan: Optional[str] = None
    plan_days: Optional[List[List[DailyTask]]] = None

    approved: Optional[bool] = None
    review_feedback: Optional[str] = None
    paused: bool = False

    calendar_events: Optional[List[dict]] = None
    notifications: Optional[List[str]] = None


class EnhancedIntent(BaseModel):
    skill: str = Field(..., description="The single primary skill the user wants to develop.")
    daily_minutes: int = Field(..., ge=5, le=300, description="Minutes per day.")
    duration_days: int = Field(30, ge=7, le=365, description="Program length in days.")
    clarified_prompt: str = Field(..., description="Clean version of the user's goal.")

    @field_validator("skill")
    @classmethod
    def tidy_skill(cls, v: str) -> str:
        return v.strip()


class SupervisorDecision(BaseModel):
    next: Literal["enhancer", "researcher", "planner", "reviewer", "executor"] = Field(...)
    reason: str = Field(...)


class ResearchFindings(BaseModel):
    resources: List[Resource] = Field(...)


class PlanDraft(BaseModel):
    days: List[List[DailyTask]] = Field(...)
    summary: str = Field(...)


class ReviewDecision(BaseModel):
    approved: bool = Field(..., description="Whether the user approved the plan.")
    feedback: str = Field(..., description="Requested changes if not approved.")
