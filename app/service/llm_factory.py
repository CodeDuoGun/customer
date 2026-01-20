"""
LLM Factory - Bridge to new modular LLM architecture.
This maintains backward compatibility while using the new LLM factory system.
"""
from customer.llm_factory.factory import llm_factory as new_llm_factory
from customer.utils.logger import logger


class LLMFactory:
    """Legacy LLM Factory - now delegates to new modular system."""

    def __init__(self):
        """Initialize LLM factory."""
        self._llm_instances = {}

    def create_llm(self, model_name: str):
        """
        Create LLM instance based on model name.

        Args:
            model_name: Name of the model to create

        Returns:
            LLM instance with legacy interface
        """
        try:
            # Use new factory to create LLM
            llm = new_llm_factory.create_llm(model_name)

            # Extend with legacy methods for backward compatibility
            self._extend_legacy_methods(llm)

            logger.info(f"Created legacy-compatible LLM instance for model: {model_name}")
            return llm

        except Exception as e:
            logger.error(f"Failed to create LLM instance for {model_name}: {str(e)}")
            # Return a basic mock LLM for testing
            return MockLLM()

    def _extend_legacy_methods(self, llm):
        """Extend LLM with legacy methods for backward compatibility."""

        # The new LLM classes already have the required methods
        # No additional extension needed
        pass

    def get_available_models(self):
        """Get available models."""
        return new_llm_factory.get_available_models()

    def get_model_info(self, model_name: str):
        """Get model information."""
        return new_llm_factory.get_model_info(model_name)


class MockLLM:
    """Mock LLM for testing when real LLM is not available."""

    def __init__(self):
        self.api_key = "mock"
        self.base_url = ""
        self.model_name = "mock"

    def call_intent_stream(self, messages, **kwargs):
        """Mock intent detection."""
        json_schema = kwargs.get('json_schema')
        if json_schema and 'intent' in json_schema.get('properties', {}):
            return {"intent": "other"}
        return "mock response"

    def chat_coze_stream(self, sse_data, messages, history, conversation_id, trace_id, query, **kwargs):
        """Mock streaming chat."""
        import json

        mock_content = "这是来自Mock LLM的回复，用于测试目的。"
        sse_data["content"] = mock_content
        sse_data["event"] = "completed"

        yield f"event: conversation\n\ndata: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
        yield "event: [DONE]\ndata: [DONE]\n\n"

    def init_client(self, api_key: str, base_url: str = ""):
        """Mock client initialization."""
        pass


# Global LLM factory instance
llm_factory = LLMFactory()
