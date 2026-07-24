"""MLflow model-from-code entrypoint.

This file is useful if you use MLflow's model-from-code style. The pyfunc wrapper
is the most generic route for custom model serving, but this file is kept so you
can switch to LangChain/LangGraph flavor logging in Databricks if desired.
"""

import mlflow

from self_help_agent.graph import build_graph

app = build_graph(interrupt_after_reviewer=False)
mlflow.models.set_model(app)
