-- Starter Lakehouse monitoring / feedback tables.
-- Adjust catalog/schema names to your environment.

CREATE SCHEMA IF NOT EXISTS main.agent_monitoring;

CREATE TABLE IF NOT EXISTS main.agent_monitoring.self_help_agent_feedback (
  request_id STRING,
  thread_id STRING,
  agent_version STRING,
  user_input STRING,
  final_output STRING,
  approval_status BOOLEAN,
  user_feedback STRING,
  latency_ms DOUBLE,
  token_count DOUBLE,
  safety_score DOUBLE,
  plan_quality_score DOUBLE,
  created_at TIMESTAMP
)
USING DELTA;

CREATE TABLE IF NOT EXISTS main.agent_monitoring.self_help_agent_eval_results (
  eval_run_id STRING,
  model_name STRING,
  model_version STRING,
  git_sha STRING,
  metric_name STRING,
  metric_value DOUBLE,
  passed BOOLEAN,
  created_at TIMESTAMP
)
USING DELTA;
