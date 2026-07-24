from __future__ import annotations

from typing import Any

import pandas as pd

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None

from self_help_agent.graph import run_until_review, run_with_approval, state_to_dict


if mlflow is not None:
    BasePythonModel = mlflow.pyfunc.PythonModel
else:  # pragma: no cover
    class BasePythonModel:  # type: ignore
        pass


class SelfHelpAgentPyFuncModel(BasePythonModel):
    """Stateless MLflow pyfunc wrapper for Mosaic AI Model Serving.

    Expected serving input can be either a pandas DataFrame with columns:
    - user_text: str
    - approval_text: optional str. If supplied, the graph runs through executor.

    Or a dict/list shape depending on serving client behavior.
    """

    def predict(self, context: Any, model_input: Any, params: dict | None = None):
        rows = _normalize_rows(model_input)
        outputs: list[dict[str, Any]] = []

        for row in rows:
            user_text = row.get("user_text") or row.get("input") or row.get("query")
            approval_text = row.get("approval_text")

            if not user_text:
                outputs.append({"error": "Missing user_text/input/query"})
                continue

            if approval_text:
                state = run_with_approval(str(user_text), str(approval_text))
            else:
                state = run_until_review(str(user_text))

            outputs.append(state_to_dict(state))

        return outputs


def _normalize_rows(model_input: Any) -> list[dict[str, Any]]:
    if isinstance(model_input, pd.DataFrame):
        return model_input.to_dict(orient="records")

    if isinstance(model_input, list):
        return [x if isinstance(x, dict) else {"input": x} for x in model_input]

    if isinstance(model_input, dict):
        if "dataframe_records" in model_input:
            return model_input["dataframe_records"]
        if "instances" in model_input:
            return model_input["instances"]
        return [model_input]

    return [{"input": str(model_input)}]
