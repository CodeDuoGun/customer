"""
LLM Factory for creating different LLM instances.
"""
from typing import Optional
from customer.config.config import config
from customer.llm_factory.base_llm import BaseLLM
from customer.llm_factory.openai_llm import OpenAILLM
from customer.llm_factory.doubao_llm import DoubaoLLM
from customer.llm_factory.qwen_llm import QwenLLM
from customer.llm_factory.llama_llm import LlamaLLM
from customer.utils.logger import logger


class LLMFactory:
    """Factory class for creating LLM instances."""

    def __init__(self):
        """Initialize LLM factory."""
        self._llm_instances = {}

    def create_llm(self, model_name: str) -> BaseLLM:
        """
        Create LLM instance based on model name.

        Args:
            model_name: Name of the model to create

        Returns:
            LLM instance
        """
        try:
            # Create cache key
            cache_key = model_name

            if cache_key in self._llm_instances:
                return self._llm_instances[cache_key]

            # Determine LLM type and create instance
            llm = self._create_llm_instance(model_name)

            # Initialize client based on model type
            self._init_llm_client(llm, model_name)

            # Cache the instance
            self._llm_instances[cache_key] = llm

            logger.info(f"Created LLM instance: {llm}")
            return llm

        except Exception as e:
            logger.error(f"Failed to create LLM instance for {model_name}: {str(e)}")
            # Return a mock LLM for testing
            return self._create_mock_llm(model_name)

    def _create_llm_instance(self, model_name: str) -> BaseLLM:
        """Create specific LLM instance based on model name."""
        # Doubao models
        if "doubao" in model_name.lower():
            return DoubaoLLM(model_name=model_name)

        # Qwen models
        elif "qwen" in model_name.lower():
            return QwenLLM(model_name=model_name)

        # Llama models
        elif "llama" in model_name.lower() or "meta/" in model_name.lower():
            return LlamaLLM(model_name=model_name)

        # DeepSeek models (OpenAI compatible)
        elif "deepseek" in model_name.lower():
            return OpenAILLM(model_name=model_name)

        # Coze models (OpenAI compatible)
        elif "coze" in model_name.lower():
            return OpenAILLM(model_name=model_name)

        # Default to OpenAI
        else:
            return OpenAILLM(model_name=model_name)

    def _init_llm_client(self, llm: BaseLLM, model_name: str) -> None:
        """Initialize LLM client with appropriate credentials."""
        try:
            # Doubao models
            if "doubao" in model_name.lower():
                llm.init_client(
                    api_key=config.ark_api_key,
                    base_url=config.ark_base_url
                )

            # Qwen models
            elif "qwen" in model_name.lower():
                llm.init_client(
                    api_key=config.qwen_api_key or config.ark_api_key,
                    base_url=config.qwen_base_url or config.ark_base_url
                )

            # Llama models
            elif "llama" in model_name.lower() or "meta/" in model_name.lower():
                llm.init_client(
                    api_key=config.replicate_api_key or "",
                    base_url=""
                )

            # DeepSeek models
            elif "deepseek" in model_name.lower():
                llm.init_client(
                    api_key=config.deepseek_api_key,
                    base_url=config.deepseek_api_url
                )

            # Coze models
            elif "coze" in model_name.lower():
                llm.init_client(
                    api_key=config.ark_api_key,
                    base_url=config.ark_base_url
                )

            # Default OpenAI
            else:
                llm.init_client(
                    api_key=config.openai_api_key or config.ark_api_key,
                    base_url=config.openai_base_url or config.ark_base_url
                )

        except Exception as e:
            logger.warning(f"Failed to initialize client for {model_name}: {str(e)}")
            # Continue without client initialization - will use mock if needed

    def _create_mock_llm(self, model_name: str) -> BaseLLM:
        """Create a mock LLM for testing when real LLM is not available."""
        logger.warning(f"Creating mock LLM for {model_name}")

        class MockLLM(BaseLLM):
            def init_client(self, api_key: str, base_url: str = "") -> None:
                pass

            def call(self, messages, **kwargs) -> str:
                return "这是来自Mock LLM的回复，用于测试目的。"

            def call_stream(self, messages, **kwargs):
                mock_response = "这是来自Mock LLM的流式回复，用于测试目的。"
                for char in mock_response:
                    yield char

            def call_intent_stream(self, messages, **kwargs):
                json_schema = kwargs.get("json_schema")
                if json_schema and 'intent' in json_schema.get('properties', {}):
                    return {"intent": "other"}
                return "mock response"

        return MockLLM(api_key="mock", base_url="", model_name=model_name)

    def get_available_models(self) -> list[str]:
        """Get list of available model names."""
        return [
            # Doubao models
            "doubao-lite-32k",
            "doubao-lite-128k",
            "doubao-pro-32k",
            "doubao-pro-128k",

            # Qwen models
            "qwen-turbo",
            "qwen-plus",
            "qwen-max",
            "qwen-max-longcontext",

            # Llama models
            "meta/llama-2-7b-chat",
            "meta/llama-2-13b-chat",
            "meta/llama-2-70b-chat",

            # DeepSeek models
            "deepseek-chat",
            "deepseek-coder",

            # Coze models
            "coze_deepseek",
            "coze_deepseek-r1",

            # OpenAI models
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo"
        ]

    def get_model_info(self, model_name: str) -> dict:
        """Get information about a specific model."""
        model_info = {
            "doubao-lite-32k": {
                "provider": "Doubao",
                "context_length": 32768,
                "supports_streaming": True,
                "supports_json_schema": True,
                "description": "Doubao Lite 32K model"
            },
            "qwen-turbo": {
                "provider": "Qwen",
                "context_length": 8192,
                "supports_streaming": True,
                "supports_function_calling": True,
                "description": "Qwen Turbo model"
            },
            "meta/llama-2-7b-chat": {
                "provider": "Meta",
                "context_length": 4096,
                "supports_streaming": True,
                "supports_json_schema": False,
                "description": "Llama 2 7B Chat model"
            },
            "deepseek-chat": {
                "provider": "DeepSeek",
                "context_length": 32768,
                "supports_streaming": True,
                "supports_json_schema": True,
                "description": "DeepSeek Chat model"
            },
            "gpt-4": {
                "provider": "OpenAI",
                "context_length": 8192,
                "supports_streaming": True,
                "supports_json_schema": True,
                "description": "GPT-4 model"
            }
        }

        return model_info.get(model_name, {
            "provider": "Unknown",
            "context_length": 4096,
            "supports_streaming": True,
            "supports_json_schema": False,
            "description": f"{model_name} model"
        })

    def clear_cache(self) -> None:
        """Clear the LLM instance cache."""
        self._llm_instances.clear()
        logger.info("Cleared LLM instance cache")


# Global LLM factory instance
llm_factory = LLMFactory()
