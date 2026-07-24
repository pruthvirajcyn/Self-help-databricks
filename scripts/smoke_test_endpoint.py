from __future__ import annotations

import argparse
import os

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    host = os.environ["DATABRICKS_HOST"].rstrip("/")
    token = os.environ["DATABRICKS_TOKEN"]

    payload = {
        "dataframe_records": [
            {
                "user_text": "I want to improve public speaking. I can do 30 minutes daily for 14 days.",
                "approval_text": "yes",
            }
        ]
    }

    response = requests.post(
        f"{host}/serving-endpoints/{args.endpoint_name}/invocations",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=args.timeout,
    )
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()
