"""Unit test for LLMs

Run this test script with:

`uv run pytest tests/test_llms.py -rA --log-level=DEBUG`
"""

import pytest
import logging

from gpt_trainer_sdk import get_available_models

logger = logging.getLogger(__name__)

def test_get_available_models():
    models = get_available_models()
    logger.info(f"Available models: {models}")
    assert isinstance(models, list)
    assert len(models) > 0
    assert all(isinstance(model, str) for model in models)