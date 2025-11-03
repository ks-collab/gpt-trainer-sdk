import pytest
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

from gpt_trainer_sdk import GPTTrainer, Chatbot

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def gpt_trainer() -> GPTTrainer:
    load_dotenv()
    base_url = os.getenv("GPT_TRAINER_API_URL", "https://app.gpt-trainer.com")
    logger.info(f"Initialized GPT-trainer client for {base_url}")
    gpt_trainer = GPTTrainer(
        api_key=os.getenv("GPT_TRAINER_API_KEY", ""),
        base_url=base_url,
        verify_ssl=(
            False if "localhost" in os.getenv("GPT_TRAINER_API_URL", "") else True
        ),
    )
    yield gpt_trainer


@pytest.fixture(scope="function")
def chatbot(gpt_trainer: GPTTrainer, request: pytest.FixtureRequest) -> Chatbot:
    chatbot_name = f"{request.node.name}-{datetime.now().strftime("%Y%m%d%H%M%S")}"
    chatbot = gpt_trainer.create_chatbot(chatbot_name)
    logger.info(f"Created chatbot {chatbot_name} with uuid {chatbot.uuid}")
    yield chatbot
    resp = gpt_trainer.delete_chatbot(chatbot.uuid)
    logger.info(f"Deleted chatbot {chatbot_name} with uuid {chatbot.uuid}")
    logger.debug(f"delete_chatbot response: {resp}")
