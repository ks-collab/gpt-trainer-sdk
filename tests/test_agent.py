"""Integration test for agent API (no costs incurred)

Run this test script with:

`uv run pytest tests/test_agent.py -rA --log-level=DEBUG`
"""

import pytest
import logging
from time import sleep
import io

from gpt_trainer_sdk import GPTTrainer, AgentUpdateOptions, Chatbot

logger = logging.getLogger(__name__)


def test_update_agent_model(gpt_trainer: GPTTrainer, chatbot: Chatbot):
    agents = gpt_trainer.get_agents(chatbot.uuid)
    updated_agent = gpt_trainer.update_agent(
        agents[0].uuid,
        AgentUpdateOptions(
            model="gpt-5-16k",
        ),
    )
    assert updated_agent.meta.model == "gpt-5-16k", "Expected model to be updated"


def test_update_agent_data_source(gpt_trainer: GPTTrainer, chatbot: Chatbot):

    # upload docs
    doc_1 = gpt_trainer.upload_data_source(
        chatbot.uuid,
        io.StringIO("Yesterday, Alice and Bob talked about pizza."),
        "test1.txt",
    )
    doc_2 = gpt_trainer.upload_data_source(
        chatbot.uuid,
        io.StringIO("Yesterday, Alice and Bob talked about ramen."),
        "test2.txt",
    )
    for try_num in range(6):
        data_sources = gpt_trainer.get_data_sources(chatbot.uuid)
        if all(data_source.status == "success" for data_source in data_sources):
            break
        sleep(5)
    else:
        logger.warning(f"documents: {data_sources}")
        raise Exception(f"Expected documents to be ready after {try_num} tries")
    data_sources = gpt_trainer.get_data_sources(chatbot.uuid)

    # use first doc
    agents = gpt_trainer.get_agents(chatbot.uuid)
    updated_agent = gpt_trainer.update_agent(
        agents[0].uuid,
        AgentUpdateOptions(
            data_source_uuids=[doc_1.uuid],
            use_all_sources=False,
            # use a fast, cheap model
            model="gpt-5-nano-16k"
        ),
    )
    assert updated_agent.data_source_uuids == [
        doc_1.uuid
    ], "Expected data source to be updated"
    assert updated_agent.meta.use_all_sources is False, "Expected use_all_sources to be False"

    # check with message
    session1 = gpt_trainer.create_chat_session(chatbot.uuid)
    message = gpt_trainer.send_message(session1.uuid, "What did Bob and Alice talk about yesterday?")
    logger.info(f"message: {message.response}")
    assert "pizza" in message.response, "Expected message to contain pizza"

    # use second doc
    updated_agent = gpt_trainer.update_agent(
        agents[0].uuid,
        AgentUpdateOptions(
            data_source_uuids=[doc_2.uuid],
        ),
    )
    assert updated_agent.data_source_uuids == [doc_2.uuid], "Expected data source to be updated"        
    assert updated_agent.meta.use_all_sources is False, "Expected use_all_sources to be False"
    
    # check with message
    session2 = gpt_trainer.create_chat_session(chatbot.uuid)
    message = gpt_trainer.send_message(session2.uuid, "What did Bob and Alice talk about yesterday?")
    logger.info(f"message: {message.response}")
    assert "ramen" in message.response, "Expected message to contain ramen"

    # revert to both docs
    updated_agent = gpt_trainer.update_agent(
        agents[0].uuid,
        AgentUpdateOptions(
            use_all_sources=True,
        ),
    )
    assert updated_agent.meta.use_all_sources is True, "Expected use_all_sources to be True"
    
    # check with message
    session3 = gpt_trainer.create_chat_session(chatbot.uuid)
    message = gpt_trainer.send_message(session3.uuid, "What did Bob and Alice talk about yesterday?")
    logger.info(f"message: {message.response}")
    assert "pizza" in message.response, "Expected message to contain pizza"
    assert "ramen" in message.response, "Expected message to contain ramen"
