"""Integration test for LLMs (no costs incurred)

Run this test script with:

`uv run pytest tests/test_llms.py -rA --log-level=DEBUG`
"""

import pytest
import logging
import os

from dotenv import load_dotenv

from gpt_trainer_sdk import GPTTrainer

logger = logging.getLogger(__name__)


load_dotenv()
base_url = os.getenv("GPT_TRAINER_API_URL", "https://app.gpt-trainer.com")
logger.info(f"Running tests against {base_url}")
gpt_trainer = GPTTrainer(
    api_key=os.getenv("GPT_TRAINER_API_KEY", ""),
    base_url=base_url,
    verify_ssl=False if "localhost" in os.getenv("GPT_TRAINER_API_URL", "") else True,
)


def test_fetch_agent_models():
    agent_models = gpt_trainer.agent_models
    logger.info(f"Agent models: {agent_models}")
    assert isinstance(agent_models, list)
    assert len(agent_models) > 0
    assert all(isinstance(model, str) for model in agent_models)


def test_fetch_agent_model_costs():
    agent_model_costs = gpt_trainer.agent_model_costs
    logger.info(f"Agent model costs: {agent_model_costs}")
    assert isinstance(agent_model_costs, dict)
    assert len(agent_model_costs) > 0
    assert all(isinstance(model, str) for model in agent_model_costs)


def test_is_valid_model():
    assert gpt_trainer.is_valid_model("gpt-4o-8k")
    assert not gpt_trainer.is_valid_model("gpt-4o-mini")


def test_model_cost():
    assert gpt_trainer.model_cost("gpt-5-8k") == 10
    assert gpt_trainer.model_cost("claude-4.5-sonnet-128k") == 264
