from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from eval.custom_scorers import (
    approval_classification_correct,
    executor_event_count_matches_duration,
    plan_has_daily_tasks,
)
from self_help_agent.config import DEFAULT_CONFIG
from self_help_agent.graph import run_with_approval, state_to_dict
from self_help_agent.model_wrapper import SelfHelpAgentPyFuncModel

try:
    from mlflow.genai.scorers import Guidelines, RelevanceToQuery, Safety
except Exception:  # pragma: no cover
    Guidelines = None
    RelevanceToQuery = None
    Safety = None


def load_eval_dataset(path: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            inputs = record.get("inputs", {})
            expectations = record.get("expectations", {})
            rows.append({**inputs, **expectations})
    return pd.DataFrame(rows)


def predict_agent_for_eval(row: dict | pd.Series) -> dict[str, Any]:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    user_text = row.get("input") or row.get("user_text")
    if not user_text:
        return {"error": "missing input"}
    # Evaluate through executor by simulating approval.
    state = run_with_approval(str(user_text), "yes")
    return state_to_dict(state)


def build_scorers():
    scorers = [plan_has_daily_tasks, executor_event_count_matches_duration, approval_classification_correct]
    if RelevanceToQuery is not None:
        scorers.append(RelevanceToQuery())
    if Safety is not None:
        scorers.append(Safety())
    if Guidelines is not None:
        scorers.append(
            Guidelines(
                guidelines=[
                    "The plan must be structured and practical.",
                    "The plan must respect the user's time budget.",
                    "The response must not promise guaranteed outcomes.",
                    "For financial or trading skills, include educational framing and risk awareness.",
                ]
            )
        )
    return scorers


def metric_value(metrics: dict[str, Any], possible_names: list[str]) -> float | None:
    for name in possible_names:
        if name in metrics and isinstance(metrics[name], (int, float)):
            return float(metrics[name])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--eval-dataset", default="eval/eval_dataset.jsonl")
    parser.add_argument("--register-on-pass", default="false")
    args = parser.parse_args()

    register_on_pass = args.register_on_pass.lower() == "true"
    uc_model_name = os.getenv("UC_MODEL_NAME", DEFAULT_CONFIG.uc_model_name)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_CONFIG.mlflow_experiment_name)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc"))
    mlflow.set_experiment(experiment_name)

    try:
        import mlflow.langchain
        mlflow.langchain.autolog()
    except Exception:
        pass

    eval_df = load_eval_dataset(args.eval_dataset)

    with mlflow.start_run(run_name=f"ci-self-help-agent-{args.git_sha}") as run:
        run_id = run.info.run_id
        mlflow.log_param("git_sha", args.git_sha)
        mlflow.log_param("branch", args.branch)
        mlflow.log_param("agent_framework", "LangGraph")
        mlflow.log_param("llm_model", DEFAULT_CONFIG.llm_model)
        mlflow.log_param("temperature", DEFAULT_CONFIG.temperature)
        mlflow.log_param("tavily_max_results", DEFAULT_CONFIG.tavily_max_results)
        mlflow.log_param("uc_model_name", uc_model_name)

        # Snapshot source/prompt files for reproducibility.
        for file_path in Path("src/self_help_agent").glob("*.py"):
            mlflow.log_artifact(str(file_path), artifact_path="source_snapshot")
        mlflow.log_artifact(args.eval_dataset, artifact_path="eval")

        input_example = pd.DataFrame(
            [{"user_text": "I want to improve public speaking. I can do 40 minutes daily for 8 weeks.", "approval_text": "yes"}]
        )

        model_info = mlflow.pyfunc.log_model(
            artifact_path="self_help_agent",
            python_model=SelfHelpAgentPyFuncModel(),
            input_example=input_example,
            code_paths=["src"],
            pip_requirements=[
                "mlflow[databricks]>=3.1.0",
                "langgraph>=0.2.70",
                "langchain>=0.3.0",
                "langchain-openai>=0.3.0",
                "langchain-tavily>=0.1.0",
                "pydantic>=2.7.0",
                "pandas>=2.2.0",
            ],
        )
        mlflow.log_param("candidate_model_uri", model_info.model_uri)

        results = mlflow.genai.evaluate(
            data=eval_df,
            predict_fn=predict_agent_for_eval,
            scorers=build_scorers(),
        )

        metrics = getattr(results, "metrics", {}) or {}
        numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)
        mlflow.log_dict(metrics, "eval/eval_metrics.json")

        print("Evaluation metrics:")
        print(json.dumps(metrics, indent=2, default=str))

        # Keep gate names tolerant because MLflow scorer metric keys vary by version.
        gates = {
            "plan_has_daily_tasks": (
                ["plan_has_daily_tasks/mean", "plan_has_daily_tasks"],
                0.80,
            ),
            "executor_event_count_matches_duration": (
                [
                    "executor_event_count_matches_duration/mean",
                    "executor_event_count_matches_duration",
                ],
                0.80,
            ),
        }

        failures: list[str] = []
        for gate_name, (metric_names, threshold) in gates.items():
            actual = metric_value(metrics, metric_names)
            if actual is None:
                failures.append(f"{gate_name}: missing metric, check eval output keys")
            elif actual < threshold:
                failures.append(f"{gate_name}: {actual:.3f} < {threshold:.3f}")

        if failures:
            mlflow.set_tag("ci_status", "failed")
            mlflow.log_text("\n".join(failures), "eval/failed_gates.txt")
            raise SystemExit("Quality gates failed:\n" + "\n".join(failures))

        mlflow.set_tag("ci_status", "passed")

        if register_on_pass:
            registered = mlflow.register_model(model_info.model_uri, uc_model_name)
            client = MlflowClient(registry_uri=os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc"))
            client.set_registered_model_alias(
                name=uc_model_name,
                alias="candidate",
                version=registered.version,
            )
            client.set_model_version_tag(uc_model_name, registered.version, "git_sha", args.git_sha)
            client.set_model_version_tag(uc_model_name, registered.version, "ci_eval_run_id", run_id)
            print(f"Registered {uc_model_name} version={registered.version} as @candidate")


if __name__ == "__main__":
    main()
