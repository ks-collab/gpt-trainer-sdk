"""Integration test of the SDK basic functions

run this test script with `poetry run python -m tests.eetest_gpt_trainer_sdk`
"""

import logging
from time import sleep
import os

from dotenv import load_dotenv

from gpt_trainer_sdk import GPTTrainer, AgentUpdateOptions, GPTTrainerError

logging.basicConfig(
    level=logging.DEBUG, format="%(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
gpt_trainer = GPTTrainer(
    api_key=os.getenv("GPT_TRAINER_API_KEY", ""),
    base_url=os.getenv("GPT_TRAINER_API_URL", ""),
)

# delete previous testing chatbots
names_to_delete = ["test-chatbot"]
chatbots = gpt_trainer.get_chatbots()
chatbots_to_delete = [
    chatbot for chatbot in chatbots if chatbot.name in names_to_delete
]
for chatbot in chatbots_to_delete:
    resp = gpt_trainer.delete_chatbot(chatbot.uuid)
    logger.info(f"Deleted chatbot {chatbot.name} with uuid {chatbot.uuid} - {resp}")


chatbot = gpt_trainer.create_chatbot("test-chatbot")

# modify agent
agents = gpt_trainer.get_agents(chatbot.uuid)
resp = gpt_trainer.update_agent(
    agents[0].uuid,
    AgentUpdateOptions(
        name="Test Agent Name",
        description="You are a test agent",
        prompt="hello world!",
        model="gpt-4o-8k",
    ),
)
agents = gpt_trainer.get_agents(chatbot.uuid)

# upload a document with unsupported file type
logger.info("uploading file with unsupported file type")
temp_file_unsupported = "expect_failure.foobar"
with open(temp_file_unsupported, "w") as f:
    f.write("Yesterday, Alice and Bob talked about their favorite pizza restaurants.")
with open(temp_file_unsupported, "rb") as f:
    try:
        upload_response_unsupported = gpt_trainer.upload_data_source(
            chatbot.uuid, f, temp_file_unsupported
        )
        assert False, "Expected an exception for unsupported file type"
    except GPTTrainerError as e:
        logger.info(f"Expected error: {e}")
        assert "file type not allowed" in str(e)
os.remove(temp_file_unsupported)

# upload documents
logger.info("uploading file")
temp_file_name = "test.txt"
with open(temp_file_name, "w") as f:
    f.write("Yesterday, Alice and Bob talked about their favorite pizza restaurants.")
with open(temp_file_name, "rb") as f:
    upload_response = gpt_trainer.upload_data_source(chatbot.uuid, f, temp_file_name)
logger.info(upload_response)
os.remove(temp_file_name)

# check document status
logger.info("checking document status")
data_sources = gpt_trainer.get_data_sources(chatbot.uuid)
while data_sources[0].status != "success":
    logger.info("document is not ready yet, trying again")
    sleep(5)
    data_sources = gpt_trainer.get_data_sources(chatbot.uuid)

logger.info(data_sources)

# send message
session = gpt_trainer.create_chat_session(chatbot.uuid)
message = gpt_trainer.send_message(session.uuid, "What did Alice talk about yesterday?")

logger.info(chatbot)
logger.info(session)
logger.info(message)

assert "pizza" in message.response

# delete data source
gpt_trainer.delete_data_source(data_sources[0].uuid)

# delete chatbot
gpt_trainer.delete_chatbot(chatbot.uuid)
