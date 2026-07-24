# Self-Help LangGraph Agent on Databricks

This repo is an end-to-end scaffold for a Databricks-oriented Agentic AI project:

- LangGraph multi-agent runtime
- MLflow tracking, tracing, model logging, GenAI evaluation, and Unity Catalog registration
- Unity Catalog governed model naming and prompt registration
- Mosaic AI Model Serving deployment script
- GitHub Actions CI/CD pipeline
- Starter tests and custom agent evaluation scorers

The agent flow is:

```text
User Request
  -> Supervisor
  -> Enhancer
  -> Researcher
  -> Planner
  -> Reviewer
  -> Approval
  -> Executor
```

The production lifecycle is:

```text
GitHub Actions
  -> Unit / graph / schema tests
  -> MLflow candidate run + prompt/model logging
  -> Agent Evaluation quality gate
  -> Unity Catalog model registration / alias
  -> Mosaic AI Model Serving endpoint update
  -> Lakehouse Monitoring / production traces
  -> Agent Evaluation feedback loop
```

## What you must configure

Set these environment variables locally or in GitHub Actions secrets:

```bash
export DATABRICKS_HOST="https://<workspace-url>"
export DATABRICKS_TOKEN="<token-or-oidc-config>"
export MLFLOW_TRACKING_URI="databricks"
export MLFLOW_REGISTRY_URI="databricks-uc"
export OPENAI_API_KEY="<your-key>"
export TAVILY_API_KEY="<your-key>"
export UC_MODEL_NAME="main.ai_agents.self_help_langgraph_agent"
export MLFLOW_EXPERIMENT_NAME="/Shared/self_help_agent_ci"
```

## Local smoke test

```bash
pip install -r requirements.txt
pytest -q
python -m self_help_agent.graph "I want to improve public speaking. I can do 40 minutes daily for 8 weeks."
```

## CI/CD flow

The GitHub Actions workflow does this:

1. Installs dependencies.
2. Runs unit tests.
3. Registers prompt versions in MLflow Prompt Registry.
4. Logs the candidate agent model to MLflow.
5. Runs MLflow GenAI evaluation and custom scorers.
6. Fails the build if quality gates fail.
7. Registers the passing model in Unity Catalog on `main`.
8. Deploys/updates the Mosaic AI Model Serving endpoint on `main`.

## Important notes

This is a scaffold, not a drop-in enterprise deployment. Workspace policies, serving endpoint permissions, enabled Databricks features, package versions, and Unity Catalog privileges differ by org. Validate the bundle and endpoint scripts in your workspace before production use.

For production, prefer service principals / GitHub OIDC over long-lived PATs, and keep secrets in Databricks Secrets or your enterprise secret manager.
