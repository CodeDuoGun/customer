"""
Doubao (豆包) LLM implementation.
"""
from typing import List, Dict, Any, Union, Generator
import json
from customer.llm_factory.openai_llm import OpenAILLM
from customer.utils.logger import logger
from customer.config.config import config


class DoubaoLLM(OpenAILLM):
    """Doubao LLM implementation, extends OpenAI LLM."""

    def __init__(self, api_key: str = config.ARK_API_KEY, base_url: str = config.ARK_BASE_URL, model_name: str = config.DOUBAO_TEXT_MODEL):
        """
        Initialize Doubao LLM.

        Args:
            api_key: Doubao API key
            base_url: Base URL for API
            model_name: Model name (e.g., "doubao-lite-32k", "doubao-pro-32k")
        """
        super().__init__(api_key, base_url, model_name)

    def init_client(self, api_key: str, base_url: str = "") -> None:
        """Initialize Doubao client."""
        # Doubao uses OpenAI-compatible API
        super().init_client(api_key, base_url)

        logger.info(f"Initialized Doubao client for model: {self.model_name}")

    def preprocess_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess messages for Doubao specific requirements."""
        processed = []

        for msg in messages:
            # Doubao may have specific message format requirements
            processed_msg = msg.copy()

            # Ensure content is string
            if isinstance(msg.get("content"), list):
                # Convert list content to string (for multimodal inputs)
                processed_msg["content"] = str(msg["content"])
            elif not isinstance(msg.get("content"), str):
                processed_msg["content"] = str(msg.get("content", ""))

            processed.append(processed_msg)

        return processed

    def call_intent_stream(self, messages: List[Dict[str, Any]], **kwargs) -> Union[str, Dict[str, Any]]:
        """Enhanced intent detection for Doubao."""
        # Doubao is good at structured output
        kwargs.setdefault("temperature", 0.1)  # Lower temperature for more consistent results
        return super().call_intent_stream(messages, **kwargs)

    @property
    def max_context_length(self) -> int:
        """Doubao model context lengths."""
        model_context_lengths = {
            "doubao-lite-32k": 32768,
            "doubao-lite-128k": 131072,
            "doubao-pro-32k": 32768,
            "doubao-pro-128k": 131072,
        }
        return model_context_lengths.get(self.model_name, 32768)

    def _get_model_specific_params(self, **kwargs) -> Dict[str, Any]:
        """Get Doubao-specific parameters."""
        params = {}

        # Doubao supports specific parameters
        if self.model_name.startswith("doubao-pro"):
            # Pro models support more advanced features
            params["top_p"] = kwargs.get("top_p", 0.7)
        else:
            # Lite models are more conservative
            params["top_p"] = kwargs.get("top_p", 0.9)

        return params
