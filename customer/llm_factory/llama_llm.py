"""
Llama LLM implementation using Replicate or other services.
"""
from typing import List, Dict, Any, Union, Generator
import json
from customer.llm_factory.base_llm import BaseLLM
from customer.utils.logger import logger


class LlamaLLM(BaseLLM):
    """Llama LLM implementation."""

    def __init__(self, api_key: str = "", base_url: str = "", model_name: str = "meta/llama-2-7b-chat"):
        """
        Initialize Llama LLM.

        Args:
            api_key: API key (Replicate token)
            base_url: Base URL (not used for Replicate)
            model_name: Model name (e.g., "meta/llama-2-7b-chat", "meta/llama-2-13b-chat")
        """
        super().__init__(api_key, base_url, model_name)
        self._client = None

    def init_client(self, api_key: str, base_url: str = "") -> None:
        """Initialize Llama client (Replicate)."""
        try:
            # Try to import replicate
            try:
                import replicate
            except ImportError:
                logger.warning("Replicate package not installed, using mock client")
                self._client = MockReplicateClient(api_key)
                return

            self.api_key = api_key
            replicate.Client(api_token=self.api_key)
            self._client = replicate

            logger.info(f"Initialized Llama client for model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Llama client: {str(e)}")
            raise

    def call(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Call Llama LLM (non-streaming)."""
        if not self._client:
            raise ValueError("Client not initialized")

        try:
            # Convert messages to Llama format
            prompt = self._messages_to_llama_prompt(messages)

            # Prepare parameters
            call_params = {
                "prompt": prompt,
                "model": self.model_name,
                "temperature": kwargs.get("temperature", 0.7),
                "max_length": kwargs.get("max_tokens", 500),
                "top_p": kwargs.get("top_p", 1),
                "repetition_penalty": 1.15,
            }

            if hasattr(self._client, 'run'):
                # Using replicate.run
                output = self._client.run(self.model_name, input=call_params)
                result = "".join(output)
            else:
                # Mock response
                result = "这是来自Mock Llama模型的回复，用于测试目的。"

            return self.postprocess_response(result)

        except Exception as e:
            logger.error(f"Llama call failed: {str(e)}")
            raise

    def call_stream(self, messages: List[Dict[str, Any]], **kwargs) -> Generator[str, None, None]:
        """Call Llama LLM with streaming."""
        if not self._client:
            raise ValueError("Client not initialized")

        try:
            prompt = self._messages_to_llama_prompt(messages)

            call_params = {
                "prompt": prompt,
                "model": self.model_name,
                "temperature": kwargs.get("temperature", 0.7),
                "max_length": kwargs.get("max_tokens", 500),
                "top_p": kwargs.get("top_p", 1),
            }

            if hasattr(self._client, 'run'):
                # Using replicate.run with streaming
                output = self._client.run(self.model_name, input=call_params)

                for item in output:
                    if isinstance(item, str):
                        yield item
                    else:
                        yield str(item)
            else:
                # Mock streaming response
                mock_response = "这是来自Mock Llama模型的流式回复，用于测试目的。"
                for char in mock_response:
                    yield char

        except Exception as e:
            logger.error(f"Llama streaming call failed: {str(e)}")
            raise

    def call_intent_stream(self, messages: List[Dict[str, Any]], **kwargs) -> Union[str, Dict[str, Any]]:
        """Call LLM for intent detection."""
        # Llama models are less structured, use basic implementation
        try:
            response_text = self.call(messages, temperature=0.1, max_tokens=500)

            json_schema = kwargs.get("json_schema")
            if json_schema:
                try:
                    result = json.loads(response_text)
                    return result
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON response from Llama")
                    return response_text
            else:
                return response_text

        except Exception as e:
            logger.error(f"Llama intent detection failed: {str(e)}")
            return {}

    def _messages_to_llama_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert chat messages to Llama prompt format."""
        prompt_parts = []

        system_message = ""
        conversation_history = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system_message = content
            elif role == "user":
                conversation_history.append(f"Human: {content}")
            elif role == "assistant":
                conversation_history.append(f"Assistant: {content}")

        # Build final prompt
        if system_message:
            prompt_parts.append(f"System: {system_message}")

        prompt_parts.extend(conversation_history)
        prompt_parts.append("Assistant:")  # Prompt for completion

        return "\n\n".join(prompt_parts)

    @property
    def supports_json_schema(self) -> bool:
        """Llama has limited JSON schema support."""
        return False

    @property
    def max_context_length(self) -> int:
        """Llama model context lengths."""
        model_context_lengths = {
            "meta/llama-2-7b-chat": 4096,
            "meta/llama-2-13b-chat": 4096,
            "meta/llama-2-70b-chat": 4096,
        }
        return model_context_lengths.get(self.model_name, 4096)


class MockReplicateClient:
    """Mock Replicate client for testing."""

    def __init__(self, api_token: str):
        self.api_token = api_token

    def run(self, model_name: str, input: Dict[str, Any]):
        """Mock run method."""
        return ["这是来自Mock Llama模型的回复，用于测试目的。"]
