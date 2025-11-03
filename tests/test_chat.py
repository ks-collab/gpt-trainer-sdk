"""test chat features

uv run pytest tests/test_chat.py -rA --log-level=DEBUG
"""

import pytest
import logging
import json

from gpt_trainer_sdk import GPTTrainer, Chatbot, AgentUpdateOptions

logger = logging.getLogger(__name__)


def test_chat_token_info(gpt_trainer: GPTTrainer, chatbot: Chatbot):
    # set model to an OpenAI model
    agents = gpt_trainer.get_agents(chatbot.uuid)
    gpt_trainer.update_agent(
        agents[0].uuid,
        AgentUpdateOptions(
            # use a fast, cheap model
            model="gpt-5-nano-16k"
        ),
    )

    # send any message
    session = gpt_trainer.create_chat_session(chatbot.uuid)
    message = gpt_trainer.send_message(session.uuid, "What is AI?")
    messages = gpt_trainer.get_messages(session.uuid)
    logger.info(f"messages: {messages}")

    last_message = messages[-1]
    last_message_meta = json.loads(last_message.meta_json)
    input_tokens = last_message_meta["actual_token_distribution"]["total_tokens"]
    logger.info(f"last_message_meta: {last_message_meta}")

    # assert that token info is in chat message response
    assert (
        "actual_token_distribution" in last_message_meta
    ), "Expected actual_token_distribution to be in meta_json"
    assert input_tokens > 0, "Expected input tokens to be greater than 0"
