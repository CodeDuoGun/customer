"""
Qwen (千问) LLM implementation.
"""
from typing import List, Dict, Any, Union, Generator
import json
from customer.llm_factory.openai_llm import OpenAILLM
from customer.utils.logger import logger


class QwenLLM(OpenAILLM):
    """Qwen LLM implementation, extends OpenAI LLM."""

    def __init__(self, api_key: str = "", base_url: str = "", model_name: str = "qwen-turbo"):
        """
        Initialize Qwen LLM.

        Args:
            api_key: Qwen API key
            base_url: Base URL for API
            model_name: Model name (e.g., "qwen-turbo", "qwen-plus", "qwen-max")
        """
        super().__init__(api_key, base_url, model_name)

    def init_client(self, api_key: str, base_url: str = "") -> None:
        """Initialize Qwen client."""
        # Qwen uses OpenAI-compatible API
        super().init_client(api_key, base_url)

        logger.info(f"Initialized Qwen client for model: {self.model_name}")

    def preprocess_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess messages for Qwen specific requirements."""
        processed = []

        for msg in messages:
            processed_msg = msg.copy()

            # Qwen supports tool calls and function calling
            if msg.get("role") == "tool":
                # Handle tool call results
                processed_msg["role"] = "tool"
                if "tool_call_id" in msg:
                    processed_msg["tool_call_id"] = msg["tool_call_id"]

            processed.append(processed_msg)

        return processed

    def call_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], **kwargs) -> str:
        """Call Qwen with tool/function calling support."""
        if not self._client:
            raise ValueError("Client not initialized")

        try:
            processed_messages = self.preprocess_messages(messages)

            call_params = {
                "model": self.model_name,
                "messages": processed_messages,
                "tools": tools,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000),
            }

            response = self._client.chat.completions.create(**call_params)
            result = response.choices[0].message.content

            return self.postprocess_response(result)

        except Exception as e:
            logger.error(f"Qwen tool call failed: {str(e)}")
            raise

    @property
    def supports_function_calling(self) -> bool:
        """Qwen supports function calling."""
        return True

    @property
    def max_context_length(self) -> int:
        """Qwen model context lengths."""
        model_context_lengths = {
            "qwen-turbo": 8192,
            "qwen-plus": 32768,
            "qwen-max": 8192,
            "qwen-max-longcontext": 32768,
        }
        return model_context_lengths.get(self.model_name, 8192)

    def _optimize_for_qwen(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Qwen-specific optimizations."""
        params = {}

        # Qwen models work well with specific temperature ranges
        if self.model_name == "qwen-max":
            params["temperature"] = min(kwargs.get("temperature", 0.7), 0.8)  # Cap temperature
        elif self.model_name == "qwen-turbo":
            params["temperature"] = max(kwargs.get("temperature", 0.7), 0.3)  # Minimum temperature

        return params
