import logging
from typing import Literal, BinaryIO, Optional, Iterator, Any
from datetime import datetime
import json
from functools import cached_property
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pydantic import BaseModel


__all__ = [
    "GPTTrainer",
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


class Chatbot(BaseModel):
    uuid: str
    name: str
    meta: dict
    created_at: str
    modified_at: str


class ChatSession(BaseModel):
    uuid: str
    created_at: str
    modified_at: str


class SendMessageResponse(BaseModel):
    response: str


class DeleteAgentResponse(BaseModel):
    success: bool


class ChatMessageCitation(BaseModel):
    data_source_uuid: str
    title: str
    text: str
    file_name: str
    type: str


ChatMessageCitations = dict[str, ChatMessageCitation]


class ChatMessage(BaseModel):
    background_pending_tasks: int
    cite_data_json: str
    cite_data: ChatMessageCitations
    created_at: datetime
    detected_frustrations: Optional[str] = None
    error_message: str
    feedback_json: Optional[str] = None
    finish_reason: str
    labels: list
    meta_json: str
    modified_at: datetime
    query: str
    response: str
    uuid: str
    session_documents: list
    ai_context_json: Optional[str] = "{}"


class DataSource(BaseModel):
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


class DataSourceFull(DataSource):
    created_at: datetime
    modified_at: datetime
    file_size: int
    meta_json: str
    tokens: int


class AgentMeta(BaseModel):
    model: str
    temperature: float
    use_all_sources: Optional[bool] = None


class Agent(BaseModel):
    created_at: datetime
    description: str
    enabled: int
    meta: AgentMeta
    modified_at: datetime
    name: str
    prompt: str
    type: str
    uuid: str
    variables: list = []
    data_source_uuids: Optional[list[str]] = None
    tool_functions: Optional[list] = None


class AgentCreateOptions(BaseModel):
    name: str
    type: Literal[
        "user-facing", "background", "human-escalation", "pre-canned", "spam-defense"
    ]
    description: str | None = None
    prompt: str | None = None
    model: str | None = None
    temperature: float | None = None
    bias: float | None = None
    stickness: float | None = None
    default_message: str | None = None
    tags: list | None = None
    use_all_sources: bool | None = None


class AgentUpdateOptions(BaseModel):
    name: Optional[str] = None
    prompt: Optional[str] = None
    description: Optional[str] = None
    model: Optional[str] = None
    enabled: Optional[bool] = None
    data_source_uuids: Optional[list[str]] = None
    temerature: Optional[float] = None
    use_all_sources: Optional[bool] = None


class SourceTag(BaseModel):
    uuid: str
    name: str
    color: str
    data_source_uuids: list[str]
    created_at: str
    modified_at: str


class SourceTagCreateOptions(BaseModel):
    name: str
    color: str
    data_source_uuids: list[str] | None = None


class SourceTagUpdateOptions(BaseModel):
    name: str | None = None
    color: str | None = None
    data_source_uuids: list[str] | None = None


class DeleteSourceTagResponse(BaseModel):
    success: bool


class ToolFunctionMeta(BaseModel):
    allow_gpt_trainer_properties: Optional[int] = None


class ToolFunction(BaseModel):
    uuid: str
    name: str
    description: str
    parameters: dict[str, Any]
    headers: dict[str, Any]
    enabled: int
    external_url: Optional[str]
    method: Optional[str]
    created_at: str
    modified_at: str
    meta: ToolFunctionMeta


class CreateToolFunctionOptions(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = {}
    headers: dict[str, Any] = {}
    enabled: int = 1
    external_url: str
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET"
    allow_gpt_trainer_properties: int = 0


class UpdateToolFunctionOptions(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None
    headers: Optional[dict[str, Any]] = None
    enabled: Optional[int] = None
    external_url: Optional[str] = None
    method: Optional[Literal["GET", "POST", "PUT", "DELETE", "PATCH"]] = None
    allow_gpt_trainer_properties: Optional[int] = None


class DeleteToolFunctionResponse(BaseModel):
    success: bool


class GPTTrainerError(Exception):
    """Raised when the GPT-trainer API returns an error response"""

    pass


class _LoggingRetry(Retry):
    """Retry strategy that logs a warning on each retry attempt."""

    def increment(
        self,
        method=None,
        url=None,
        response=None,
        error=None,
        _pool=None,
        _stacktrace=None,
    ):
        if response is not None:
            path = urlparse(url).path if url else "?"
            attempt = len(self.history) + 1
            max_retries = self.total if isinstance(self.total, int) else 3
            logger.warning(
                "Retrying request to %s %s (attempt %d/%d) after %s %s",
                method or "?",
                path,
                attempt,
                max_retries,
                response.status,
                response.reason or "",
            )
        elif error is not None:
            path = urlparse(url).path if url else "?"
            attempt = len(self.history) + 1
            max_retries = self.total if isinstance(self.total, int) else 3
            logger.warning(
                "Retrying request to %s %s (attempt %d/%d) after connection error: %s",
                method or "?",
                path,
                attempt,
                max_retries,
                error,
            )
        return super().increment(
            method=method,
            url=url,
            response=response,
            error=error,
            _pool=_pool,
            _stacktrace=_stacktrace,
        )


def _create_retry_adapter(
    max_retries: int = 3,
    backoff_factor: float = 2,
) -> HTTPAdapter:
    """Create an HTTPAdapter with retry logic for transient failures."""
    retry_strategy = _LoggingRetry(
        total=max_retries,
        status_forcelist=[502, 503, 504],
        allowed_methods=None,  # Retry all methods; duplicate risk is low for 502/503/504
        backoff_factor=backoff_factor,
        raise_on_status=False,
    )
    return HTTPAdapter(max_retries=retry_strategy)


class GPTTrainer:

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://app.gpt-trainer.com",
        verify_ssl: bool = True,
        max_retries: int = 3,
        backoff_factor: float = 2,
        timeout: float = 120,
    ):
        if not verify_ssl:
            logger.warning("SSL verification is disabled")

        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
        }
        self.api_url = f"{self.base_url}/api/v1"
        self.verify_ssl = verify_ssl
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.verify = verify_ssl
        self.session.mount(
            "https://",
            _create_retry_adapter(max_retries, backoff_factor),
        )
        self.session.mount(
            "http://",
            _create_retry_adapter(max_retries, backoff_factor),
        )

        logger.debug(f"Initialized GPTTrainer with base URL {self.base_url}")

    def _request(self, method: str, url: str, timeout=None, **kwargs):
        """Make an HTTP request with retry and timeout. Converts transport failures to GPTTrainerError."""
        if timeout is None:
            timeout = self.timeout
        try:
            return self.session.request(method, url, timeout=timeout, **kwargs)
        except requests.exceptions.RequestException as e:
            raise GPTTrainerError(f"Request failed: {e}") from e

    def get_chatbots(self) -> list[Chatbot]:
        url = f"{self.api_url}/chatbots"
        response = self._request("GET", url)

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
        response = self._request("POST", url, json=data)

        if response.status_code == 200:
            logger.debug(f"Chatbot created - {response.text}")
            return Chatbot(**response.json())
        else:
            raise GPTTrainerError(
                f"Failed to create chatbot - HTTP {response.status_code}: {response.text}"
            )

    def update_chatbot(self, chatbot_uuid: str, options: dict[str, Any]) -> Chatbot:
        url = f"{self.api_url}/chatbot/{chatbot_uuid}/update"
        response = self._request("POST", url, json=options)
        if response.status_code == 200:
            logger.debug(f"Chatbot updated - {response.text}")
            return Chatbot(**response.json())
        else:
            raise GPTTrainerError(
                f"Failed to update chatbot - HTTP {response.status_code}: {response.text}"
            )

    def delete_chatbot(self, chatbot_uuid: str):
        url = f"{self.api_url}/chatbot/{chatbot_uuid}/delete"
        response = self._request("DELETE", url)

        if response.status_code == 200:
            logger.debug(f"Chatbot {chatbot_uuid} deleted - {response.text}")
        else:
            raise GPTTrainerError(
                f"Failed to delete chatbot {chatbot_uuid} - HTTP {response.status_code}: {response.text}"
            )

    def create_chat_session(self, chatbot_uuid: str) -> ChatSession:
        url = f"{self.api_url}/chatbot/{chatbot_uuid}/session/create"
        response = self._request("POST", url)

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
        response = self._request("POST", url, json=payload)

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
        # Use tuple timeout: (connect_timeout, read_timeout) for streaming
        response = self._request(
            "POST",
            url,
            json={"query": query},
            stream=True,
            timeout=(10, self.timeout),
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
        response = self._request("GET", url)

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

        response = self._request("POST", url, files=files, data=payload)

        if response.status_code == 200:
            logger.debug(f"File upload successful - {response.text}")
            return DataSource(**response.json())
        elif response.status_code == 409:
            logger.debug(f"File already exists - {response.text}")
            return DataSource(**response.json())
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

        response = self._request("GET", url)

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
        response = self._request("POST", url)

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

        response = self._request("POST", url, json=payload)

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

        response = self._request("GET", url)

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
        response = self._request("POST", url, json=options_dict)
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

        response = self._request("POST", url, json=options_dict)

        if response.status_code == 200:
            logger.debug(f"Updated agent {agent_uuid} - {response.text}")
            return Agent(**response.json())
        else:
            raise GPTTrainerError(
                f"Failed to update agent {agent_uuid} - HTTP {response.status_code}: {response.text}"
            )

    def delete_agent(self, agent_uuid: str) -> DeleteAgentResponse:
        url = f"{self.api_url}/agent/{agent_uuid}/delete"
        response = self._request("DELETE", url)
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
        response = self._request("POST", url, json=options_dict)

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
        response = self._request("GET", url)

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
        response = self._request("POST", url, json=options_dict)

        if response.status_code == 200:
            logger.debug(f"Updated source tag {source_tag_uuid} - {response.text}")
            return SourceTag(**response.json())
        else:
            raise GPTTrainerError(
                f"Failed to update source tag {source_tag_uuid} - HTTP {response.status_code}: {response.text}"
            )

    def delete_source_tag(self, source_tag_uuid: str) -> DeleteSourceTagResponse:
        url = f"{self.api_url}/source-tag/{source_tag_uuid}/delete"
        response = self._request("DELETE", url)

        if response.status_code == 200:
            logger.debug(f"Deleted source tag {source_tag_uuid} - {response.text}")
            return DeleteSourceTagResponse(**response.json())
        else:
            raise GPTTrainerError(
                f"Failed to delete source tag {source_tag_uuid} - HTTP {response.status_code}: {response.text}"
            )

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

        response = self._request(
            "POST",
            f"{self.api_url}/session-document/upload",
            files=files,
            data=data,
        )

        if response.status_code == 200:
            logger.debug(f"Session document uploaded - {response.text}")
            return response.json()
        else:
            raise GPTTrainerError(
                f"Failed to upload session document - HTTP {response.status_code}: {response.text}"
            )

    def create_tool_function(
        self, agent_uuid: str, options: CreateToolFunctionOptions
    ) -> ToolFunction:
        """Create a tool function for an agent.

        Args:
            agent_uuid: The UUID of the agent to create the tool function for
            options: The options for the tool function

        Returns:
            The tool function object
        """
        options_dict = {k: v for k, v in options.__dict__.items() if v is not None}
        response = self._request(
            "POST",
            f"{self.api_url}/agent/{agent_uuid}/function/create",
            json=options_dict,
        )
        if response.status_code == 200:
            logger.debug(
                f"Tool function created for agent {agent_uuid} - {response.text}"
            )
            return ToolFunction(**response.json())
        else:
            raise GPTTrainerError(
                f"Failed to create tool function for agent {agent_uuid} - HTTP {response.status_code}: {response.text}"
            )

    def update_tool_function(
        self, tool_function_uuid: str, options: UpdateToolFunctionOptions
    ) -> ToolFunction:
        url = f"{self.api_url}/function/{tool_function_uuid}/update"
        options_dict = {k: v for k, v in options.__dict__.items() if v is not None}
        response = self._request("POST", url, json=options_dict)
        if response.status_code == 200:
            logger.debug(
                f"Tool function updated {tool_function_uuid} - {response.text}"
            )
            return ToolFunction(**response.json())
        else:
            raise GPTTrainerError(
                f"Failed to update tool function {tool_function_uuid} - HTTP {response.status_code}: {response.text}"
            )

    def delete_tool_function(
        self, tool_function_uuid: str
    ) -> DeleteToolFunctionResponse:
        url = f"{self.api_url}/function/{tool_function_uuid}/delete"
        response = self._request("POST", url)
        print(response.text)
        if response.status_code == 200:
            logger.debug(
                f"Tool function deleted {tool_function_uuid} - {response.text}"
            )
            return DeleteToolFunctionResponse(**response.json())
        else:
            raise GPTTrainerError(
                f"Failed to delete tool function {tool_function_uuid} - HTTP {response.status_code}: {response.text}"
            )

    @cached_property
    def agent_models(self) -> list[str]:
        url = f"{self.api_url}/agent/models"
        response = self._request("GET", url)
        if response.status_code == 200:
            logger.debug(f"Fetched agent models - {response.text}")
            models: list[str] = response.json()["models"]
            return models
        else:
            raise GPTTrainerError(
                f"Failed to get agent models - HTTP {response.status_code}: {response.text}"
            )

    @cached_property
    def agent_model_costs(self) -> dict[str, int]:
        url = f"{self.api_url}/agent/model-costs"
        response = self._request("GET", url)
        if response.status_code == 200:
            logger.debug(f"Fetched agent model costs - {response.text}")
            model_costs: dict[str, int] = response.json()["model_costs"]
            return model_costs
        else:
            raise GPTTrainerError(
                f"Failed to get agent model costs - HTTP {response.status_code}: {response.text}"
            )

    def is_valid_model(self, model: str) -> bool:
        return model in self.agent_models

    def model_cost(self, model: str) -> int:
        if model not in self.agent_model_costs:
            raise ValueError(f"Invalid model: {model}")
        return self.agent_model_costs[model]
