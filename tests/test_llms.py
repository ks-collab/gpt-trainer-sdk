"""Integration test for LLMs 

Run this test script with:

(no costs incurred)
`uv run pytest tests/test_llms.py -rA --log-level=DEBUG`

or 

(incurs costs)
`uv run pytest tests/test_llms.py -rA --log-level=DEBUG -m incur_costs`
"""

import pytest
import requests
import logging
import os

from dotenv import load_dotenv

from gpt_trainer_sdk import GPTTrainer, Chatbot, AgentUpdateOptions, GPTTrainerError

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


@pytest.mark.incur_costs
def test_llms_are_valid(gpt_trainer: GPTTrainer, chatbot: Chatbot):
    """Test that all models are valid and can answer a question"""
    
    # get models to test
    response = requests.get(f"{base_url}/chatbot/model-metadata", verify=False)
    model_metadata = response.json()["models"]
    smallest_models: list[str] = []
    visited_models: list[str] = []
    for model_name, model_data in model_metadata.items():
        # model must be 2k or bigger to be user-facing agent
        if model_data["name"] not in visited_models and model_data["context_size"] >= 2000:
            smallest_models.append(model_name)
            visited_models.append(model_data["name"])
    logger.info(f"Models to test: {smallest_models}")

    # test each model
    failed_models: list[str] = []
    for model_name in smallest_models:
        # set agent model to the smallest model
        logger.info(f"Setting agent model to {model_name}")
        agents = gpt_trainer.get_agents(chatbot.uuid)
        updated_agent = gpt_trainer.update_agent(
            agents[0].uuid,
            AgentUpdateOptions(
                model=model_name,
            ),
        )
        assert updated_agent.meta.model == model_name, "Expected model to be updated"

        # send message to agent
        logger.info(f"Sending message to {model_name}")
        session = gpt_trainer.create_chat_session(chatbot.uuid)
        try:
            message = gpt_trainer.send_message(session.uuid, "Hello!")
            if "currently overloaded" in message.response:
                failed_models.append(model_name)
                logger.error(f"Model {model_name} failed")
                continue
        except GPTTrainerError as e:
            failed_models.append(model_name)
            logger.error(f"Failed to send message to {model_name}: {e}")

    logger.info(f"Failed models: {failed_models}")
    assert len(failed_models) == 0, f"Some models failed: {failed_models}"
