import pandas as pd

from self_help_agent.model_wrapper import _normalize_rows


def test_normalize_dataframe_rows():
    df = pd.DataFrame([{"user_text": "hello"}])
    rows = _normalize_rows(df)
    assert rows == [{"user_text": "hello"}]


def test_normalize_dataframe_records_payload():
    rows = _normalize_rows({"dataframe_records": [{"user_text": "hello"}]})
    assert rows == [{"user_text": "hello"}]
