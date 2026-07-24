from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    llm_model: str = os.getenv("AGENT_LLM_MODEL", "gpt-4.1-nano")
    temperature: float = float(os.getenv("AGENT_TEMPERATURE", "0"))
    tavily_max_results: int = int(os.getenv("TAVILY_MAX_RESULTS", "10"))
    mlflow_experiment_name: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "/Shared/self_help_agent_ci",
    )
    uc_model_name: str = os.getenv(
        "UC_MODEL_NAME",
        "main.ai_agents.self_help_langgraph_agent",
    )


DEFAULT_CONFIG = AgentConfig()
