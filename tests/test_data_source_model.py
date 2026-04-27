"""Unit tests for the DataSource / DataSourceFull Pydantic models.

These tests run by default (no `incur_costs` marker) and don't hit the API.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from gpt_trainer_sdk import DataSource, DataSourceFull


def _full_payload(**overrides):
    base = {
        "uuid": "ds-uuid",
        "file_name": "doc.pdf",
        "title": "doc.pdf",
        "status": "success",
        "type": "upload",
        "created_at": datetime.now(),
        "modified_at": datetime.now(),
        "file_size": 1234,
        "meta_json": "{}",
        "tokens": 42,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "status",
    [
        "await",
        "queued",
        "pending",
        "downloading",
        "extracting",
        "chunking",
        "embedding",
        "success",
        "fail",
        "error:storage",
        "error:token",
        "error:rate limit",
        "error:no text",
        "error:image upload limit",
        "error:invalid data source",
        "error:invalid chatbot",
        "error:invalid user",
        "error:download",
        "error:content type",
        "error:not found",
        "error:scraper rate limit",
        "error:something brand new",
    ],
)
def test_data_source_accepts_known_and_arbitrary_error_statuses(status):
    ds = DataSourceFull(**_full_payload(status=status))
    assert ds.status == status


def test_data_source_invalid_type_still_rejected():
    with pytest.raises(ValidationError):
        DataSourceFull(**_full_payload(type="not-a-real-type"))


def test_data_source_minimal_fields():
    ds = DataSource(
        uuid="x", file_name="f", title="t", status="error:rate limit", type="upload"
    )
    assert ds.status == "error:rate limit"
