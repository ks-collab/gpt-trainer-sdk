"""Integration test for LLMs 

Run this test script with:

(no costs incurred)
`uv run pytest tests/test_llms.py -rA --log-level=DEBUG`

or 

(incurs costs)
`uv run pytest tests/test_llms.py -rA --log-level=DEBUG -m incur_costs`

Options for the incur_costs test (environment variables):
- LLM_TEST_EXCLUDE_PREFIXES: comma-separated model-name prefixes to skip. Defaults to
  "groq-": Groq models are listed by /chatbot/model-metadata but hidden in the app UI
  when no platform Groq key is installed, so they always fail. Set to "" to test them.
- LLM_TEST_MODELS: comma-separated substrings; only test models containing one of them,
  e.g. LLM_TEST_MODELS=gpt-5.6,claude-5 to check newly added models cheaply.
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

TEST_QUERY = "Hello!"
EXCLUDED_MODEL_PREFIXES = tuple(
    p.strip() for p in os.getenv("LLM_TEST_EXCLUDE_PREFIXES", "groq-").split(",") if p.strip()
)
ONLY_MODELS = tuple(
    s.strip() for s in os.getenv("LLM_TEST_MODELS", "").split(",") if s.strip()
)
# Reply text the backend sends in place of an answer when the LLM call fails
ERROR_REPLY_MARKERS = (
    "An error occurred while processing your message",
    "currently overloaded",
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
    skipped = [m for m in smallest_models if m.startswith(EXCLUDED_MODEL_PREFIXES)]
    smallest_models = [m for m in smallest_models if m not in skipped]
    if ONLY_MODELS:
        smallest_models = [m for m in smallest_models if any(s in m for s in ONLY_MODELS)]
    if skipped:
        logger.info(f"Skipping models (LLM_TEST_EXCLUDE_PREFIXES): {skipped}")
    logger.info(f"Models to test: {smallest_models}")
    assert smallest_models, "No models selected to test"

    # test each model
    failed_models: dict[str, str] = {}
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
            message = gpt_trainer.send_message(session.uuid, TEST_QUERY)
            reason = _failure_reason(gpt_trainer, session.uuid, message.response)
        except GPTTrainerError as e:
            reason = f"request failed: {e}"
        if reason:
            failed_models[model_name] = reason
            logger.error(f"Model {model_name} failed: {reason}")
        else:
            logger.info(f"Model {model_name} OK: {message.response.strip()[:80]!r}")

    logger.info(f"Failed models: {failed_models}")
    assert not failed_models, "Some models failed:\n" + "\n".join(
        f"  {model}: {reason}" for model, reason in failed_models.items()
    )


def _failure_reason(gpt_trainer: GPTTrainer, session_uuid: str, reply: str) -> str | None:
    """Return why a reply counts as a failure, or None if the model answered.

    The non-stream endpoint returns HTTP 200 even when the LLM call fails: the reply is
    the chatbot's error message ("An error occurred while processing your message...")
    and the stored message has finish_reason "error" plus a traceback in error_message.
    """
    if not reply.strip():
        return "empty reply"
    for message in gpt_trainer.get_messages(session_uuid):
        if message.query == TEST_QUERY and (
            message.finish_reason == "error" or message.error_message.strip()
        ):
            last_line = (message.error_message.strip().splitlines() or [""])[-1]
            return f"finish_reason={message.finish_reason}: {last_line[:200]}"
    # Fallback in case the stored message isn't available
    for marker in ERROR_REPLY_MARKERS:
        if marker.lower() in reply.lower():
            return f"error reply: {reply.strip()[:120]!r}"
    return None
