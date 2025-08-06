"""Integration test of the SDK basic functions

1. set up .env
2. run this test script with `uv run python -m tests.eetest_gpt_trainer_sdk`
"""

import logging
from time import sleep
import os

from dotenv import load_dotenv

from gpt_trainer_sdk import GPTTrainer, AgentUpdateOptions, GPTTrainerError, AgentCreateOptions, SourceTagCreateOptions, SourceTagUpdateOptions

logging.basicConfig(
    level=logging.DEBUG, format="%(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
gpt_trainer = GPTTrainer(
    api_key=os.getenv("GPT_TRAINER_API_KEY", ""),
    base_url=os.getenv("GPT_TRAINER_API_URL", "https://app.gpt-trainer.com"),
    verify_ssl=False if "localhost" in os.getenv("GPT_TRAINER_API_URL", "") else True
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

# create and delete agent
# expected response to delete agent: {"success": true}
new_agent = gpt_trainer.create_agent(chatbot.uuid, AgentCreateOptions(
    name="Test Agent Name",
    type="user-facing",
    description="You are a test agent",
    prompt="You are a test agent",
))
resp = gpt_trainer.delete_agent(new_agent.uuid)
assert resp.success, "Expected success response"

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

# test source tags
logger.info("testing source tags")
source_tag = gpt_trainer.create_source_tag(chatbot.uuid, SourceTagCreateOptions(
    name="Test Tag",
    color="#FF5733",
    data_source_uuids=[data_sources[0].uuid]
))
source_tags = gpt_trainer.get_source_tags(chatbot.uuid)
assert len(source_tags) >= 1, "Expected at least one source tag"
updated_source_tag = gpt_trainer.update_source_tag(source_tag.uuid, SourceTagUpdateOptions(
    name="Updated Test Tag",
    color="#33FF57",
    data_source_uuids=[data_sources[0].uuid]
))
assert updated_source_tag.name == "Updated Test Tag", "Expected name to be updated"
assert updated_source_tag.color == "#33FF57", "Expected color to be updated"
delete_response = gpt_trainer.delete_source_tag(source_tag.uuid)
assert delete_response.success, "Expected successful deletion"

# send message
session = gpt_trainer.create_chat_session(chatbot.uuid)
message = gpt_trainer.send_message(session.uuid, "What did Alice talk about yesterday?")

logger.info(chatbot)
logger.info(session)
logger.info(message)

assert "pizza" in message.response

# send message stream
num_chunks = 0
for chunk in gpt_trainer.send_message_stream(session.uuid, "Write a 5-paragraph essay explaining what machine learning is."):
    num_chunks += 1
    logger.info(f"Streaming response chunk: {chunk}")
    # print(chunk, end='', flush=True)  # Print without newlines for smooth output
assert num_chunks >= 2, "Expected at least 2 chunks"

# get messages
messages = gpt_trainer.get_messages(session.uuid)
logger.info(messages)

# retry data source
gpt_trainer.retry_data_source(data_sources[0].uuid)

# delete data source
gpt_trainer.delete_data_source(data_sources[0].uuid)

# delete chatbot
gpt_trainer.delete_chatbot(chatbot.uuid)
