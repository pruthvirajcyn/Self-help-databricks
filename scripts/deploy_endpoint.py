from __future__ import annotations

import argparse
import os

import mlflow
from mlflow.deployments import get_deploy_client
from mlflow.tracking import MlflowClient


def get_model_version_by_alias(model_name: str, alias: str) -> str:
    client = MlflowClient(registry_uri=os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc"))
    model_version = client.get_model_version_by_alias(name=model_name, alias=alias)
    return str(model_version.version)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--model-name", default=os.getenv("UC_MODEL_NAME"))
    parser.add_argument("--alias", default="candidate")
    parser.add_argument("--workload-size", default="Small")
    parser.add_argument("--promote-alias", default="staging")
    args = parser.parse_args()

    if not args.model_name:
        raise ValueError("--model-name or UC_MODEL_NAME is required")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc"))

    model_version = get_model_version_by_alias(args.model_name, args.alias)
    deploy_client = get_deploy_client("databricks")

    served_entity_name = args.model_name.replace(".", "_") + f"_v{model_version}"
    config = {
        "served_entities": [
            {
                "name": served_entity_name,
                "entity_name": args.model_name,
                "entity_version": model_version,
                "workload_size": args.workload_size,
                "scale_to_zero_enabled": True,
            }
        ]
    }

    try:
        deploy_client.get_endpoint(endpoint=args.endpoint_name)
        deploy_client.update_endpoint_config(endpoint=args.endpoint_name, config=config)
        print(f"Updated endpoint {args.endpoint_name} to {args.model_name} v{model_version}")
    except Exception:
        deploy_client.create_endpoint(name=args.endpoint_name, config=config)
        print(f"Created endpoint {args.endpoint_name} for {args.model_name} v{model_version}")

    client = MlflowClient(registry_uri=os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc"))
    client.set_registered_model_alias(
        name=args.model_name,
        alias=args.promote_alias,
        version=model_version,
    )
    print(f"Set alias @{args.promote_alias} -> version {model_version}")


if __name__ == "__main__":
    main()
