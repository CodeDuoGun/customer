"""
Data models for chat service.
"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class ChatRequest(BaseModel):
    """Chat request model."""
    conversation_id: str
    query: str
    model_name: str = "coze_deepseek"


class CreateChatItem(BaseModel):
    """Create chat item model."""
    uid: str
    device_id: str = "shyl"


class IntentItem(BaseModel):
    """Intent detection item model."""
    conversation_id: str
    query: str


class UpdateQAItem(BaseModel):
    """Update QA item model."""
    question: str
    answer: str = ""
    new_question: str
    new_answer: str


class DoctorInfo(BaseModel):
    """Doctor information model."""
    doctor_name: str
    hospital_name: str
    specialty: str
    introduction: str = ""
    consultation_location: str = ""


class SearchResult(BaseModel):
    """Search result model."""
    content: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class ConversationMessage(BaseModel):
    """Conversation message model."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
