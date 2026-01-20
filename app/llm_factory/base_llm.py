"""
Base LLM class defining the interface for all LLM implementations.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Generator
from customer.utils.logger import logger


class BaseLLM(ABC):
    """Abstract base class for all LLM implementations."""

    def __init__(self, api_key: str = "", base_url: str = "", model_name: str = ""):
        """
        Initialize the LLM.

        Args:
            api_key: API key for the LLM service
            base_url: Base URL for the LLM service
            model_name: Name of the model to use
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self._client = None
        self._tokenizer = None

    @abstractmethod
    def init_client(self, api_key: str, base_url: str = "") -> None:
        """
        Initialize the LLM client.

        Args:
            api_key: API key
            base_url: Base URL (optional)
        """
        pass

    @abstractmethod
    def call(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """
        Call the LLM with messages (non-streaming).

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters

        Returns:
            Generated response text
        """
        pass

    @abstractmethod
    def call_stream(self, messages: List[Dict[str, Any]], **kwargs) -> Generator[str, None, None]:
        """
        Call the LLM with streaming response.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters

        Yields:
            Response chunks
        """
        pass

    @abstractmethod
    def call_intent_stream(self, messages: List[Dict[str, Any]], **kwargs) -> Union[str, Dict[str, Any]]:
        """
        Call the LLM for intent detection with structured output.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters (json_schema, etc.)

        Returns:
            Structured intent result or raw response
        """
        pass

    def chat_coze_stream(
        self,
        sse_data: Dict[str, Any],
        messages: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
        conversation_id: str,
        trace_id: str,
        query: str,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Unified streaming chat method for SSE responses.

        Args:
            sse_data: SSE data structure
            messages: Chat messages
            history: Conversation history
            conversation_id: Conversation ID
            trace_id: Trace ID for logging
            query: User query
            **kwargs: Additional parameters

        Yields:
            SSE formatted chunks
        """
        try:
            # Update SSE data with current content
            need_stream_content = kwargs.get('need_stream_content', True)
            need_update_history = kwargs.get('need_update_history', True)
            chat_model = kwargs.get('chat_model', self.model_name)
            max_tokens = kwargs.get('max_tokens', 1000)
            temperature = kwargs.get('temperature', 0.7)

            # Call streaming response
            response_chunks = self.call_stream(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            content_buffer = ""

            for chunk in response_chunks:
                content_buffer += chunk

                if need_stream_content:
                    sse_data["content"] = chunk
                    yield f"event: conversation\n\ndata: {self._format_sse_data(sse_data)}\n\n"

            # Final update
            if need_update_history and content_buffer:
                sse_data["content"] = content_buffer
                sse_data["event"] = "completed"

                if history is not None:
                    history.append({
                        "role": "assistant",
                        "content": content_buffer
                    })

                yield f"event: conversation\n\ndata: {self._format_sse_data(sse_data)}\n\n"

            yield "event: [DONE]\ndata: [DONE]\n\n"

        except Exception as e:
            logger.error(f"{trace_id}: Streaming chat failed: {str(e)}")
            sse_data["content"] = "抱歉，发生了错误，请稍后再试。"
            sse_data["event"] = "completed"
            yield f"event: conversation\n\ndata: {self._format_sse_data(sse_data)}\n\n"
            yield "event: [DONE]\ndata: [DONE]\n\n"

    def _format_sse_data(self, data: Dict[str, Any]) -> str:
        """Format data for SSE response."""
        import json
        return json.dumps(data, ensure_ascii=False)

    def set_tokenizer(self, tokenizer) -> None:
        """Set the tokenizer for the LLM."""
        self._tokenizer = tokenizer

    def get_tokenizer(self):
        """Get the tokenizer."""
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # Fallback: rough estimate
        return len(text) // 4

    def validate_messages(self, messages: List[Dict[str, Any]]) -> bool:
        """Validate message format."""
        if not isinstance(messages, list):
            return False

        required_fields = ["role", "content"]
        valid_roles = ["system", "user", "assistant", "tool"]

        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if not all(field in msg for field in required_fields):
                return False
            if msg["role"] not in valid_roles:
                return False

        return True

    def preprocess_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess messages before sending to LLM."""
        # Default: no preprocessing
        return messages

    def postprocess_response(self, response: str) -> str:
        """Postprocess response from LLM."""
        # Default: no postprocessing
        logger.info(f"model resp: {response}")
        return response.strip()

    @property
    def supports_streaming(self) -> bool:
        """Whether this LLM supports streaming."""
        return True

    @property
    def supports_json_schema(self) -> bool:
        """Whether this LLM supports JSON schema validation."""
        return False

    @property
    def max_context_length(self) -> int:
        """Maximum context length for this LLM."""
        return 4096

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(model={self.model_name})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(model={self.model_name}, api_key={'***' if self.api_key else 'None'})"
