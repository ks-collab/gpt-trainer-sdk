import logging
from dataclasses import dataclass, field
from typing import Literal, BinaryIO, Optional, Iterator
from datetime import datetime
import json
import inspect

import requests
from .llms import ModelType, is_valid_model, get_model_cost

# Export the ModelType enum for easy access
__all__ = [
    "GPTTrainer",
    "ModelType",
    "is_valid_model",
    "get_model_cost",
    "Chatbot",
    "ChatSession",
    "SendMessageResponse",
    "DeleteAgentResponse",
    "ChatMessageCitation",
    "Agent",
    "AgentCreateOptions",
    "AgentUpdateOptions",
    "SourceTag",
    "SourceTagCreateOptions",
    "SourceTagUpdateOptions",
    "DeleteSourceTagResponse",
    "GPTTrainerError",
]

logger = logging.getLogger(__name__)


@dataclass
class Chatbot:
    uuid: str
    name: str
    meta: dict
    created_at: str
    modified_at: str


@dataclass
class ChatSession:
    uuid: str
    created_at: str
    modified_at: str


@dataclass
class SendMessageResponse:
    response: str


@dataclass
class DeleteAgentResponse:
    success: bool


@dataclass
class ChatMessageCitation:
    data_source_uuid: str
    title: str
    text: str
    file_name: str
    type: str


ChatMessageCitations = dict[str, ChatMessageCitation]


@dataclass
class ChatMessage:
    background_pending_tasks: int
    cite_data_json: str
    cite_data: ChatMessageCitations
    created_at: datetime
    detected_frustrations: str
    error_message: str
    feedback_json: str
    finish_reason: str
    labels: list
    meta_json: str
    modified_at: datetime
    query: str
    response: str
    uuid: str
    session_documents: list
    ai_context_json: Optional[str] = "{}"


@dataclass
class DataSource:
    uuid: str
    file_name: str
    title: str
    status: Literal[
        "await",
        "queued",
        "pending",
        "success",
        "extracting",
        "chunking",
        "embedding",
        "error:storage",
        "error:token",
        "fail",
    ]
    type: Literal["upload", "link", "google-drive", "table", "image", "qa", "video"]

    @classmethod
    def from_dict(cls, args: dict):
        return cls(
            **{k: v for k, v in args.items() if k in inspect.signature(cls).parameters}
        )


@dataclass
class DataSourceFull(DataSource):
    created_at: datetime
    modified_at: datetime
    file_size: int
    meta_json: dict
    tokens: int


@dataclass
class AgentMeta:
    model: str
    temperature: float
    use_all_sources: bool


@dataclass
class Agent:
    created_at: datetime
    description: str
    enabled: int
    meta: AgentMeta
    modified_at: datetime
    name: str
    prompt: str
    type: Literal["user-facing"]
    uuid: str
    variables: list = field(default_factory=list)
    data_source_uuids: Optional[list[str]] = None
    tool_functions: Optional[list] = None


@dataclass
class AgentCreateOptions:
    name: str
    type: Literal[
        "user-facing", "background", "human-escalation", "pre-canned", "spam-defense"
    ]
    description: str | None = None
    prompt: str | None = None
    model: ModelType | None = None
    temperature: float | None = None
    bias: float | None = None
    stickness: float | None = None
    default_message: str | None = None
    tags: list | None = None
    use_all_sources: bool | None = None


@dataclass
class AgentUpdateOptions:
    name: str | None = None
    description: str | None = None
    prompt: str | None = None
    model: ModelType | None = None
    enabled: bool | None = None


@dataclass
class SourceTag:
    uuid: str
    name: str
    color: str
    data_source_uuids: list[str]
    created_at: str
    modified_at: str


@dataclass
class SourceTagCreateOptions:
    name: str
    color: str
    data_source_uuids: list[str] | None = None


@dataclass
class SourceTagUpdateOptions:
    name: str | None = None
    color: str | None = None
    data_source_uuids: list[str] | None = None


@dataclass
class DeleteSourceTagResponse:
    success: bool


class GPTTrainerError(Exception):
    """Raised when the GPT-trainer API returns an error response"""

    pass


class GPTTrainer:

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://app.gpt-trainer.com",
        verify_ssl: bool = True,
    ):
        if not verify_ssl:
            logger.warning("SSL verification is disabled")

        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
        }
        self.api_url = f"{self.base_url}/api/v1"
        self.verify_ssl = verify_ssl

        logger.debug(f"Initialized GPTTrainer with base URL {self.base_url}")

    def get_chatbots(self) -> list[Chatbot]:
        url = f"{self.api_url}/chatbots"
        response = requests.get(url, headers=self.headers, verify=self.verify_ssl)

        if response.status_code == 200:
            return [Chatbot(**chatbot) for chatbot in response.json()]
        else:
            raise GPTTrainerError(
                f"Failed to get chatbots - HTTP {response.status_code}: {response.text}"
            )

    def create_chatbot(self, name: str, show_citations: bool = False) -> Chatbot:
        url = f"{self.api_url}/chatbot/create"
        data = {
            "name": name,
            "rate_limit": [20, 240],
            "rate_limit_message": "Too many messages",
            "show_citations": show_citations,
            "visibility": "private",
        }
        response = requests.post(
            url, headers=self.headers, json=data, verify=self.verify_ssl
        )

        if response.status_code == 200:
            logger.debug(f"Chatbot created - {response.text}")
            return Chatbot(**response.json())
        else:
            raise GPTTrainerError(
                f"Failed to create chatbot - HTTP {response.status_code}: {response.text}"
            )

    def delete_chatbot(self, chatbot_uuid: str):
        url = f"{self.api_url}/chatbot/{chatbot_uuid}/delete"
        response = requests.delete(url, headers=self.headers, verify=self.verify_ssl)

        if response.status_code == 200:
            logger.debug(f"Chatbot {chatbot_uuid} deleted - {response.text}")
        else:
            raise GPTTrainerError(
                f"Failed to delete chatbot {chatbot_uuid} - HTTP {response.status_code}: {response.text}"
            )

    def create_chat_session(self, chatbot_uuid: str) -> ChatSession:
        url = f"{self.api_url}/chatbot/{chatbot_uuid}/session/create"
        response = requests.post(url, headers=self.headers, verify=self.verify_ssl)

        if response.status_code == 200:
            logger.debug(f"Chat session created - {response.text}")
            return ChatSession(**response.json())
        else:
            raise GPTTrainerError(
                f"Failed to create chat session - HTTP {response.status_code}: {response.text}"
            )

    def send_message(
        self,
        session_uuid: str,
        query: str,
        pre_canned_response: bool = None,
        direct_function_response: bool = None,
        session_document_uuids: list[str] = None,
    ) -> SendMessageResponse:
        payload = {
            "query": query,
        }
        if pre_canned_response is not None:
            payload["pre_canned_response"] = pre_canned_response
        if direct_function_response is not None:
            payload["direct_function_response"] = direct_function_response
        if session_document_uuids is not None:
            payload["session_document_uuids"] = session_document_uuids

        url = f"{self.api_url}/session/{session_uuid}/message/non-stream"
        logger.debug(f"Sending message with payload {payload}")
        response = requests.post(
            url,
            headers=self.headers,
            json=payload,
            verify=self.verify_ssl,
        )

        if response.status_code == 200:
            logger.debug(f"Chat message reply received - {response.text}")
            return SendMessageResponse(**response.json())
        else:
            raise GPTTrainerError(
                f"Failed to send chat message - HTTP {response.status_code}: {response.text}"
            )

    def send_message_stream(self, session_uuid: str, query: str) -> Iterator[str]:
        """Send a message and yield streaming response chunks.

        Args:
            session_uuid: The UUID of the chat session
            query: The message to send

        Yields:
            str: Individual chunks of the streaming response
        """
        url = f"{self.api_url}/session/{session_uuid}/message/stream"
        response = requests.post(
            url,
            headers=self.headers,
            json={"query": query},
            stream=True,
            verify=self.verify_ssl,
        )

        if response.status_code == 200:
            for line in response.iter_lines(decode_unicode=True):
                # Skip empty lines
                if line:
                    logger.debug(f"Streaming response chunk: {line}")
                    yield line
        else:
            raise GPTTrainerError(
                f"Failed to send streaming chat message - HTTP {response.status_code}: {response.text}"
            )

    @staticmethod
    def convert_cite_data(cite_data_json: str) -> ChatMessageCitations:
        cite_data_dict = json.loads(cite_data_json)

        if cite_data_dict:
            new_dict: ChatMessageCitations = {}
            for citation_number, citation in cite_data_dict.items():
                new_dict[citation_number] = ChatMessageCitation(**citation)
            return new_dict
        else:
            return {}

    def get_messages(self, session_uuid: str) -> list[ChatMessage]:
        url = f"{self.api_url}/session/{session_uuid}/messages"
        response = requests.get(url, headers=self.headers, verify=self.verify_ssl)

        if response.status_code == 200:
            logger.debug(f"Chat messages received - {response.text}")
            return [
                ChatMessage(
                    cite_data=GPTTrainer.convert_cite_data(message["cite_data_json"]),
                    **message,
                )
                for message in response.json()
            ]
        else:
            raise GPTTrainerError(
                f"Failed to get chat messages - HTTP {response.status_code}: {response.text}"
            )

    def upload_data_source(
        self, chatbot_uuid: str, file: BinaryIO, file_name: str
    ) -> DataSource:
        url = f"{self.api_url}/chatbot/{chatbot_uuid}/data-source/upload"
        files = {"file": (file_name, file)}

        # we don't need reference_source_link
        payload = {"reference_source_link": None}

        response = requests.post(
            url, files=files, data=payload, headers=self.headers, verify=self.verify_ssl
        )

        if response.status_code == 200:
            logger.debug(f"File upload successful - {response.text}")
            return DataSource.from_dict(response.json())
        elif response.status_code == 409:
            logger.debug(f"File already exists - {response.text}")
            return DataSource.from_dict(response.json())
        else:
            raise GPTTrainerError(
                f"Failed to upload file - HTTP {response.status_code}: {response.text}"
            )

    def upload_data_source_from_aws_s3(
        self, chatbot_uuid: str, s3_resource, bucket: str, key: str, file_name: str
    ) -> DataSource:
        """Upload a data source from an AWS S3 bucket.

        Args:
            chatbot_uuid: The UUID of the chatbot to upload the data source to
            s3_resource: The S3 resource object, e.g. with `boto3.resource("s3")`
            bucket: The name of the S3 bucket
            key: The key of the file in the S3 bucket
            file_name: The name of the file to upload

        Returns:
            The data source object
        """

        obj = s3_resource.Object(bucket, key)
        f = obj.get()["Body"].read()
        return self.upload_data_source(chatbot_uuid, f, file_name)

    def get_data_sources(self, chatbot_uuid: str) -> list[DataSourceFull]:
        url = f"{self.api_url}/chatbot/{chatbot_uuid}/data-sources"

        response = requests.get(url, headers=self.headers, verify=self.verify_ssl)

        if response.status_code == 200:
            logger.debug(f"Data sources received - {response.text}")
            return [DataSourceFull(**source) for source in response.json()]
        else:
            raise GPTTrainerError(
                f"Failed to get data sources - HTTP {response.status_code}: {response.text}"
            )

    def delete_data_source(self, data_source_uuid: str):
        """Delete a data source by its UUID.

        Args:
            data_source_uuid: The UUID of the data source to delete

        Raises:
            GPTTrainerError: If the API request fails
        """
        url = f"{self.api_url}/data-source/{data_source_uuid}/delete"
        response = requests.post(url, headers=self.headers, verify=self.verify_ssl)

        if response.status_code == 200:
            logger.debug(f"Data source {data_source_uuid} deleted - {response.text}")
        else:
            raise GPTTrainerError(
                f"Failed to delete data source {data_source_uuid} - HTTP {response.status_code}: {response.text}"
            )

    def retry_data_source(self, data_source_uuid: str):
        """Retry processing a data source by its UUID.

        Args:
            data_source_uuid: The UUID of the data source to retry

        Raises:
            GPTTrainerError: If the API request fails
        """
        url = f"{self.base_url}/api/data-sources/restart"
        payload = {"uuids": [data_source_uuid]}

        response = requests.post(
            url, headers=self.headers, json=payload, verify=self.verify_ssl
        )

        if response.status_code == 200:
            logger.debug(
                f"Data source {data_source_uuid} retry initiated - {response.text}"
            )
            return response.json()
        else:
            raise GPTTrainerError(
                f"Failed to retry data source {data_source_uuid} - HTTP {response.status_code}: {response.text}"
            )

    def get_agents(self, chatbot_uuid: str) -> list[Agent]:
        url = f"{self.api_url}/chatbot/{chatbot_uuid}/agents"

        response = requests.get(url, headers=self.headers, verify=self.verify_ssl)

        if response.status_code == 200:
            logger.debug(f"Fetched agents for chatbot {chatbot_uuid} - {response.text}")
            return [Agent(**agent) for agent in response.json()]
        else:
            raise GPTTrainerError(
                f"Failed to get agents for chatbot {chatbot_uuid} - HTTP {response.status_code}: {response.text}"
            )

    def create_agent(self, chatbot_uuid: str, options: AgentCreateOptions) -> Agent:
        url = f"{self.api_url}/chatbot/{chatbot_uuid}/agent/create"
        options_dict = {k: v for k, v in options.__dict__.items() if v is not None}
        response = requests.post(
            url, headers=self.headers, json=options_dict, verify=self.verify_ssl
        )
        if response.status_code == 200:
            logger.debug(f"Agent created for chatbot {chatbot_uuid} - {response.text}")
            return Agent(**response.json())
        else:
            raise GPTTrainerError(
                f"Failed to create agent for chatbot {chatbot_uuid} - HTTP {response.status_code}: {response.text}"
            )

    def update_agent(self, agent_uuid: str, options: AgentUpdateOptions):
        url = f"{self.api_url}/agent/{agent_uuid}/update"

        options_dict = {k: v for k, v in options.__dict__.items() if v is not None}

        response = requests.post(
            url, headers=self.headers, json=options_dict, verify=self.verify_ssl
        )

        if response.status_code == 200:
            logger.debug(f"Updated agent {agent_uuid} - {response.text}")
            return Agent(**response.json())
        else:
            raise GPTTrainerError(
                f"Failed to update agent {agent_uuid} - HTTP {response.status_code}: {response.text}"
            )

    def delete_agent(self, agent_uuid: str) -> DeleteAgentResponse:
        url = f"{self.api_url}/agent/{agent_uuid}/delete"
        response = requests.delete(url, headers=self.headers, verify=self.verify_ssl)
        if response.status_code == 200:
            logger.debug(f"Deleted agent {agent_uuid} - {response.text}")
            return DeleteAgentResponse(**response.json())
        else:
            raise GPTTrainerError(
                f"Failed to delete agent {agent_uuid} - HTTP {response.status_code}: {response.text}"
            )

    def create_source_tag(
        self, chatbot_uuid: str, options: SourceTagCreateOptions
    ) -> SourceTag:
        url = f"{self.api_url}/chatbot/{chatbot_uuid}/source-tag/create"
        options_dict = {k: v for k, v in options.__dict__.items() if v is not None}
        response = requests.post(
            url, headers=self.headers, json=options_dict, verify=self.verify_ssl
        )

        if response.status_code == 200:
            logger.debug(
                f"Source tag created for chatbot {chatbot_uuid} - {response.text}"
            )
            return SourceTag(**response.json())
        else:
            raise GPTTrainerError(
                f"Failed to create source tag for chatbot {chatbot_uuid} - HTTP {response.status_code}: {response.text}"
            )

    def get_source_tags(self, chatbot_uuid: str) -> list[SourceTag]:
        url = f"{self.api_url}/chatbot/{chatbot_uuid}/source-tags"
        response = requests.get(url, headers=self.headers, verify=self.verify_ssl)

        if response.status_code == 200:
            logger.debug(
                f"Fetched source tags for chatbot {chatbot_uuid} - {response.text}"
            )
            return [SourceTag(**tag) for tag in response.json()]
        else:
            raise GPTTrainerError(
                f"Failed to get source tags for chatbot {chatbot_uuid} - HTTP {response.status_code}: {response.text}"
            )

    def update_source_tag(
        self, source_tag_uuid: str, options: SourceTagUpdateOptions
    ) -> SourceTag:
        url = f"{self.api_url}/source-tag/{source_tag_uuid}/update"
        options_dict = {k: v for k, v in options.__dict__.items() if v is not None}
        response = requests.post(
            url, headers=self.headers, json=options_dict, verify=self.verify_ssl
        )

        if response.status_code == 200:
            logger.debug(f"Updated source tag {source_tag_uuid} - {response.text}")
            return SourceTag(**response.json())
        else:
            raise GPTTrainerError(
                f"Failed to update source tag {source_tag_uuid} - HTTP {response.status_code}: {response.text}"
            )

    def delete_source_tag(self, source_tag_uuid: str) -> DeleteSourceTagResponse:
        url = f"{self.api_url}/source-tag/{source_tag_uuid}/delete"
        response = requests.delete(url, headers=self.headers, verify=self.verify_ssl)

        if response.status_code == 200:
            logger.debug(f"Deleted source tag {source_tag_uuid} - {response.text}")
            return DeleteSourceTagResponse(**response.json())
        else:
            raise GPTTrainerError(
                f"Failed to delete source tag {source_tag_uuid} - HTTP {response.status_code}: {response.text}"
            )

    def is_model_string_valid(self, model: str) -> bool:
        """
        Check if a model string is valid.

        Args:
            model: Model name as string

        Returns:
            True if the model is valid, False otherwise
        """
        return is_valid_model(model)

    def upload_session_document(self, file: BinaryIO, filename: str) -> dict:
        """
        Upload a file as a chat session document using multipart form data.

        Args:
            file: Binary file object to upload
            filename: Filename for the uploaded file

        Returns:
            dict: Response containing the session document details
        """
        files = {"file": (filename, file)}
        data = {"filename": filename}

        response = requests.post(
            f"{self.api_url}/session-document/upload",
            files=files,
            data=data,
            headers=self.headers,
            verify=self.verify_ssl,
        )

        if response.status_code == 200:
            logger.debug(f"Session document uploaded - {response.text}")
            return response.json()
        else:
            raise GPTTrainerError(
                f"Failed to upload session document - HTTP {response.status_code}: {response.text}"
            )
