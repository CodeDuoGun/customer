"""
OpenAI-compatible LLM implementation.
Supports OpenAI API, DeepSeek, and other OpenAI-compatible services.
"""
from typing import List, Dict, Any, Union, Generator
import json

from pandas import api
from customer.llm_factory.base_llm import BaseLLM
from customer.utils.logger import logger


class OpenAILLM(BaseLLM):
    """OpenAI-compatible LLM implementation."""

    def __init__(self, api_key: str = "", base_url: str = "", model_name: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI LLM.

        Args:
            api_key: OpenAI API key
            base_url: Base URL for API (optional)
            model_name: Model name
        """
        super().__init__(api_key, base_url, model_name)
        self._client = None

    def init_client(self, api_key: str, base_url: str = "") -> None:
        """Initialize OpenAI client."""
        try:
            # Try to import openai
            try:
                from openai import OpenAI
            except ImportError:
                logger.warning("OpenAI package not installed, using mock client")
                self._client = MockOpenAIClient(api_key, base_url or self.base_url)
                return

            self.api_key = api_key
            self.base_url = base_url or self.base_url
            logger.info(f"{self.api_key}, {self.base_url}")
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=(5.0, 30)
            )

            logger.info(f"Initialized OpenAI client for model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    def call(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Call OpenAI LLM (non-streaming)."""
        if not self._client:
            raise ValueError("Client not initialized")
        logger.info(kwargs)

        try:
            # Preprocess messages
            processed_messages = self.preprocess_messages(messages)

            # Prepare parameters
            call_params = {
                "model": kwargs.get("chat_model", "deepseek-v3-250324"),
                "messages": processed_messages,
                "temperature": kwargs.get("temperature", 0.1),
                "max_tokens": kwargs.get("max_tokens", 200),
                "response_format":kwargs.get("json_schema"), 
                "top_p": kwargs.get("top_p", 1),
                "stream": False
            }

            # Add optional parameters
            if "frequency_penalty" in kwargs:
                call_params["frequency_penalty"] = kwargs["frequency_penalty"]
            if "presence_penalty" in kwargs:
                call_params["presence_penalty"] = kwargs["presence_penalty"]
            
            logger.info(f"call_params: {call_params}")
            response = self._client.chat.completions.create(**call_params)
            result = response.choices[0].message.content

            # Postprocess response
            return self.postprocess_response(result)

        except Exception as e:
            logger.error(f"OpenAI call failed: {str(e)}")
            raise

    def call_stream(self, messages: List[Dict[str, Any]], **kwargs) -> Generator[str, None, None]:
        """Call OpenAI LLM with streaming."""
        if not self._client:
            raise ValueError("Client not initialized")

        try:
            # Preprocess messages
            processed_messages = self.preprocess_messages(messages)

            # Prepare parameters
            call_params = {
                "model": kwargs.get("chat_model", "doubao-1.5-pro-32k-250115"),
                "messages": processed_messages,
                "temperature": kwargs.get("temperature", 0.1),
                "max_tokens": kwargs.get("max_tokens", 200),
                "response_format":kwargs.get("json_schema"), 
                "top_p": kwargs.get("top_p", 1),
                "stream": True
            }
            logger.info(f"call_params: {call_params}")
            response = self._client.chat.completions.create(**call_params)

            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield content

        except Exception as e:
            logger.error(f"OpenAI streaming call failed: {str(e)}")
            raise

    def call_intent_stream(self, messages: List[Dict[str, Any]], **kwargs) -> Union[str, Dict[str, Any]]:
        """Call LLM for intent detection with structured output."""
        try:
            kwargs.setdefault("chat_model", "doubao-1.5-pro-32k-250115")
            json_schema = kwargs.get("json_schema")
            stream = kwargs.get("stream", False)
            chat_model = kwargs.get("chat_model", self.model_name)
            temperature = kwargs.get("temperature", 0.1)
            max_tokens = kwargs.get("max_tokens", 500)

            # Prepare messages with JSON schema instruction if provided
            processed_messages = messages.copy()

            # Call the LLM
            if stream:
                response_text = ""
                for chunk in self.call_stream(processed_messages, json_schema=json_schema, chat_model=chat_model,temperature=temperature, max_tokens=max_tokens):
                    response_text += chunk
            else:
                response_text = self.call(processed_messages, json_schema=json_schema, chat_model=chat_model,temperature=temperature, max_tokens=max_tokens)

            # Try to parse JSON if schema was provided
            if json_schema:
                try:
                    result = json.loads(response_text)
                    return result
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON response, returning raw content")
                    return response_text
            else:
                return response_text

        except Exception as e:
            logger.error(f"Intent detection failed: {str(e)}")
            return {}

    @property
    def supports_json_schema(self) -> bool:
        """OpenAI supports structured output."""
        return True

    @property
    def max_context_length(self) -> int:
        """Maximum context length based on model."""
        model_context_lengths = {
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "deepseek-chat": 32768,
            "deepseek-coder": 16384,
        }
        return model_context_lengths.get(self.model_name, 4096)


class MockOpenAIClient:
    """Mock OpenAI client for testing when package is not available."""

    def __init__(self, api_key: str, base_url: str = ""):
        self.api_key = api_key
        self.base_url = base_url

    def chat(self):
        """Mock chat namespace."""
        return MockChatCompletions()


class MockChatCompletions:
    """Mock chat completions."""

    def create(self, **kwargs):
        """Mock create method."""
        return MockResponse()


class MockResponse:
    """Mock response object."""

    def __init__(self):
        self.choices = [MockChoice()]


class MockChoice:
    """Mock choice object."""

    def __init__(self):
        self.message = MockMessage()


class MockMessage:
    """Mock message object."""

    def __init__(self):
        self.content = "这是来自Mock OpenAI客户端的回复，用于测试目的。"
