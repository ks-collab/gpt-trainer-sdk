"""Integration test for tool functions

Run this test script with:

`uv run pytest tests/test_tool_function.py -rA --log-level=DEBUG -m incur_costs`
"""

import pytest
import logging
import json

from gpt_trainer_sdk import (
    GPTTrainer,
    AgentUpdateOptions,
    Chatbot,
    CreateToolFunctionOptions,
    UpdateToolFunctionOptions,
)

logger = logging.getLogger(__name__)


def test_tool_function_lifecycle(gpt_trainer: GPTTrainer, chatbot: Chatbot):
    # configure agent
    agents = gpt_trainer.get_agents(chatbot.uuid)
    agent_uuid = agents[0].uuid
    gpt_trainer.update_agent(
        agent_uuid,
        AgentUpdateOptions(
            name="Todos fetcher",
            prompt="Help the user fetch todos. You have access to tool functions to complete this. Interpret the user's query and call the appropriate function.",
            model="gpt-5-mini-4k",
        ),
    )

    # create tool function
    tool_function = gpt_trainer.create_tool_function(
        agent_uuid,
        CreateToolFunctionOptions(
            name="fetch_todos",
            description="placeholder",
            method="GET",
            external_url="https://jsonplaceholder.typicode.com/todos/1",
            parameters={},
        ),
    )
    assert tool_function.description == "placeholder"

    # update tool function
    tool_function = gpt_trainer.update_tool_function(
        tool_function.uuid,
        UpdateToolFunctionOptions(
            description="fetches a todo for the user",
        ),
    )
    assert tool_function.description == "fetches a todo for the user"

    # delete tool function
    gpt_trainer.delete_tool_function(tool_function.uuid)


@pytest.mark.incur_costs
def test_gemini_function_calling(gpt_trainer: GPTTrainer, chatbot: Chatbot):
    MODELS_TO_TEST = [
        "gemini-3-flash-16k",
        "gemini-3-flash-thinking-16k",
        "gemini-3-pro-16k",
        "gemini-3-pro-thinking-16k",
        "gemini-2.0-flash-128k",
        "gemini-2.0-flash-lite-128k",
        "gemini-2.5-flash-16k",
        "gemini-2.5-flash-lite-16k",
        "gemini-2.5-flash-lite-thinking-16k",
        "gemini-2.5-flash-thinking-16k",
        "gemini-2.5-pro-16k",
        "gemini-2.5-pro-thinking-16k",
    ]

    agents = gpt_trainer.get_agents(chatbot.uuid)
    agent_uuid = agents[0].uuid

    # define tool function once and reuse across models
    tool_function = gpt_trainer.create_tool_function(
        agent_uuid,
        CreateToolFunctionOptions(
            name="update_task_status",
            description="updates the status for a task for the user",
            method="PATCH",
            external_url="https://jsonplaceholder.typicode.com/posts/1",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "ID of the task",
                    },
                    "status": {
                        "type": "string",
                        "description": "Set new task status: QUEUED, PROCESSING, COMPLETED",
                    },
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["task_id", "status"],
            },
        ),
    )

    try:
        for model in MODELS_TO_TEST:
            logger.info(f"Testing Gemini function calling with model: {model}")
            gpt_trainer.update_agent(
                agent_uuid,
                AgentUpdateOptions(
                    name="Task Manager",
                    prompt="Help the user manage tasks, such as creating a task or updating the status of a task. You have access to tool functions to complete this. Interpret the user's query and call the appropriate function.",
                    model=model,
                ),
            )

            session = gpt_trainer.create_chat_session(chatbot.uuid)
            gpt_trainer.send_message(
                session.uuid, "Update task 123 to status COMPLETED"
            )
            messages = gpt_trainer.get_messages(session.uuid)
            last_message_meta = json.loads(messages[-1].meta_json)

            if last_message_meta.get("functions_called") is None:
                logger.info(f"No functions called in message {last_message_meta}")
                raise Exception(f"No functions called in message {last_message_meta}")

            logger.info(f"tool function: {last_message_meta['functions_called']}")

            assert (
                last_message_meta["functions_called"][0]["name"] == "update_task_status"
            )
            tool_function_response = json.loads(
                last_message_meta["functions_called"][0]["content"]
            )
            assert tool_function_response["task_id"] == 123
            assert tool_function_response["status"] == "COMPLETED"
    finally:
        gpt_trainer.delete_tool_function(tool_function.uuid)


@pytest.mark.incur_costs
def test_patch_verb(gpt_trainer: GPTTrainer, chatbot: Chatbot):
    # configure agent
    agents = gpt_trainer.get_agents(chatbot.uuid)
    agent_uuid = agents[0].uuid
    gpt_trainer.update_agent(
        agent_uuid,
        AgentUpdateOptions(
            name="Task Manager",
            prompt="Help the user manage tasks, such as creating a task or updating the status of a task. You have access to tool functions to complete this. Interpret the user's query and call the appropriate function.",
            model="gpt-5-mini-4k",
        ),
    )

    # define tool function
    tool_function = gpt_trainer.create_tool_function(
        agent_uuid,
        CreateToolFunctionOptions(
            name="update_task_status",
            description="updates the status for a task for the user",
            method="PATCH",
            external_url="https://jsonplaceholder.typicode.com/posts/1",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer", "description": "ID of the task"},
                    "status": {
                        "type": "string",
                        "description": "Set new task status: QUEUED, PROCESSING, COMPLETED",
                    },
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["task_id", "status"],
            },
        ),
    )

    # test message
    session = gpt_trainer.create_chat_session(chatbot.uuid)
    gpt_trainer.send_message(session.uuid, "Update task 123 to status COMPLETED")
    messages = gpt_trainer.get_messages(session.uuid)
    logger.info(f"messages: {messages}")
    last_message_meta = json.loads(messages[-1].meta_json)
    logger.info(f"tool function: {last_message_meta['functions_called']}")
    assert last_message_meta["functions_called"][0]["name"] == "update_task_status"
    tool_function_response = json.loads(
        last_message_meta["functions_called"][0]["content"]
    )
    assert tool_function_response["task_id"] == 123
    assert tool_function_response["status"] == "COMPLETED"
