from __future__ import annotations

import argparse
import os

import mlflow

from self_help_agent.prompts import (
    APPROVAL_SYSTEM,
    ENHANCER_SYSTEM,
    EXECUTOR_SYSTEM,
    PLANNER_SYSTEM,
    RESEARCHER_SYSTEM,
    REVIEWER_SYSTEM,
    SUPERVISOR_SYSTEM,
)

PROMPTS = {
    "supervisor": SUPERVISOR_SYSTEM,
    "enhancer": ENHANCER_SYSTEM,
    "researcher": RESEARCHER_SYSTEM,
    "planner": PLANNER_SYSTEM,
    "reviewer": REVIEWER_SYSTEM,
    "approval": APPROVAL_SYSTEM,
    "executor": EXECUTOR_SYSTEM,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--catalog", default=os.getenv("PROMPT_CATALOG", "main"))
    parser.add_argument("--schema", default=os.getenv("PROMPT_SCHEMA", "ai_prompts"))
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc"))

    for role, template in PROMPTS.items():
        prompt_name = f"{args.catalog}.{args.schema}.self_help_{role}"
        try:
            prompt = mlflow.genai.register_prompt(
                name=prompt_name,
                template=template,
                commit_message=f"CI prompt update from {args.branch}@{args.git_sha}",
                tags={
                    "agent": "self_help_coach",
                    "prompt_role": role,
                    "git_sha": args.git_sha,
                    "branch": args.branch,
                },
            )
            if args.branch == "main":
                mlflow.genai.set_prompt_alias(
                    name=prompt_name,
                    alias="staging",
                    version=prompt.version,
                )
            print(f"Registered prompt {prompt_name} version={prompt.version}")
        except AttributeError:
            # Older MLflow versions may not have Prompt Registry available.
            print(f"Prompt Registry unavailable; skipped {prompt_name}")


if __name__ == "__main__":
    main()
