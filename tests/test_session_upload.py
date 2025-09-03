"""Integration test for chat session file upload

1. set up .env
2. run this test script with:

 `uv run pytest tests/test_session_upload.py -rA --log-level=DEBUG -m incur_costs`
"""

import pytest
import logging
from datetime import datetime
import os
import io

from dotenv import load_dotenv

from gpt_trainer_sdk import (
    GPTTrainer,
    AgentUpdateOptions,
    ModelType,
)

logger = logging.getLogger(__name__)

load_dotenv()
gpt_trainer = GPTTrainer(
    api_key=os.getenv("GPT_TRAINER_API_KEY", ""),
    base_url=os.getenv("GPT_TRAINER_API_URL", "https://app.gpt-trainer.com"),
    verify_ssl=False if "localhost" in os.getenv("GPT_TRAINER_API_URL", "") else True,
)

TEST_CHATBOT_PREFIX = "test-chatbot"


def delete_testing_chatbots():
    # delete previous testing chatbots
    chatbots = gpt_trainer.get_chatbots()
    chatbots_to_delete = [
        chatbot for chatbot in chatbots if chatbot.name.startswith(TEST_CHATBOT_PREFIX)
    ]
    for chatbot in chatbots_to_delete:
        resp = gpt_trainer.delete_chatbot(chatbot.uuid)
        logger.info(f"Deleted chatbot {chatbot.name} with uuid {chatbot.uuid} - {resp}")


@pytest.mark.incur_costs
def test_chat_session_file_upload():
    # set up chatbot
    delete_testing_chatbots()
    chatbot = gpt_trainer.create_chatbot(
        f"{TEST_CHATBOT_PREFIX}-session_file_upload-{datetime.now().strftime("%Y%m%d%H%M%S")}"
    )

    # configure agent for file upload (need larger context)
    agents = gpt_trainer.get_agents(chatbot.uuid)
    gpt_trainer.update_agent(
        agents[0].uuid,
        AgentUpdateOptions(
            name="Test Agent Name for File Upload",
            description="You are a test agent for file upload",
            prompt="You are a test agent for file upload",
            model=ModelType.GPT_4O_MINI_32K,
        ),
    )

    # send chat message with file upload
    session = gpt_trainer.create_chat_session(chatbot.uuid)
    message = gpt_trainer.send_message(session.uuid, "hello there, what can you do?")
    logger.info(message)

    # Create a file-like object from a hardcoded string
    file_content = """The Stapler Incident

It was a typical Tuesday morning at Acme Corporation when Sarah from accounting discovered that her trusty stapler had mysteriously vanished from her desk. This wasn't just any stapler - it was the red Swingline model that had faithfully served her for three years, through countless reports, expense forms, and the occasional paper jam.

Sarah's desk neighbor, Mike from IT, noticed her frantic searching and offered his condolences. "I saw Bob from marketing walking around with a stapler yesterday," he mentioned casually while sipping his coffee. "But you know how it is around here - office supplies have a way of migrating between departments like lost socks in a dryer."

The office supply cabinet, usually stocked with pens, paper clips, and sticky notes, was surprisingly low on staplers. Sarah found only a lonely blue Bostitch model that looked like it had been through a war. "This will have to do," she muttered, testing it on a scrap piece of paper. The staple went in crooked, but it held.

Meanwhile, in the marketing department, Bob was blissfully unaware of the drama unfolding in accounting. He was using Sarah's red Swingline to staple together a presentation about Q4 projections. "This is a nice stapler," he thought to himself, admiring its smooth action and comfortable grip. "I should get one like this for my desk."

The saga continued throughout the week. Sarah's blue stapler developed a squeak that could be heard across the office. Mike suggested oiling it, but Sarah was skeptical. "It's not the same," she lamented. "My red one never made that noise."

By Friday, the missing stapler had become something of an office legend. People would ask Sarah about it in the break room, and she'd recount the tale with increasing dramatic flair. "It was the perfect stapler," she'd say, "never jammed, always aligned perfectly, and the red color matched my coffee mug."

Finally, on Monday morning, Sarah arrived at her desk to find her beloved red Swingline sitting exactly where she'd left it, with a sticky note attached: "Sorry for borrowing this! - Bob from Marketing." The note was written on a yellow Post-it that was slightly askew, as if applied in a hurry.

Sarah smiled and tested the stapler. It worked perfectly, just as she remembered. She made a mental note to label her office supplies more clearly in the future, but secretly, she was glad the incident had given her something interesting to talk about during coffee breaks.

The blue Bostitch stapler found a new home in the supply cabinet, where it would wait for the next person who needed a temporary stapling solution. And so the cycle of office supply migration continued, as it always had and always would."""
    
    # Convert string to bytes and create BytesIO object
    file_bytes = file_content.encode('utf-8')
    file_obj = io.BytesIO(file_bytes)
    
    session_document = gpt_trainer.upload_session_document(file=file_obj, filename="test_file.txt")
    session_document_uuid = session_document["uuid"]

    gpt_trainer.send_message(session.uuid, "Whose stapler vanished?", session_document_uuids=[session_document_uuid])
