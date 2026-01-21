"""
Customer service module for LlamaIndex-based chat system.
"""

__version__ = "1.0.0"
__author__ = "LlamaIndex Customer Service Team"

# Import main components for easy access
from .service.chat_service import chat_service, ChatService
from .service.intent_detector import IntentDetector
from .service.entity_extractor import EntityExtractor
from .service.doctor_search_service import DoctorSearchService
from .service.qa_search_service import QASearchService
from .service.conversation_service import conversation_service
from .config.config import config

__all__ = [
    "chat_service",
    "ChatService",
    "IntentDetector",
    "EntityExtractor",
    "DoctorSearchService",
    "QASearchService",
    "conversation_service",
    "config"
]
