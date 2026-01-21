"""
Conversation management service.
"""
from typing import List, Dict, Any, Optional
from customer.config.config import config
from customer.utils.logger import logger
from customer.schema.chat import ConversationMessage


class ConversationService:
    """Service for managing conversation history and state."""

    def __init__(self):
        """Initialize conversation service."""
        self._conversations = {}  # In-memory storage, replace with database in production

    def get_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history by ID.

        Args:
            conversation_id: Conversation ID

        Returns:
            List of conversation messages
        """
        try:
            history = self._conversations.get(conversation_id, [])
            logger.debug(f"Retrieved conversation {conversation_id}: {len(history)} messages")
            return history
        except Exception as e:
            logger.error(f"Failed to get conversation {conversation_id}: {str(e)}")
            return []

    def update_conversation(self, conversation_id: str, history: List[Dict[str, Any]], trace_id: str) -> bool:
        """
        Update conversation history.

        Args:
            conversation_id: Conversation ID
            history: Updated conversation history
            trace_id: Trace ID for logging

        Returns:
            Success status
        """
        try:
            self._conversations[conversation_id] = history
            logger.debug(f"{trace_id}: Updated conversation {conversation_id} with {len(history)} messages")
            return True
        except Exception as e:
            logger.error(f"{trace_id}: Failed to update conversation {conversation_id}: {str(e)}")
            return False

    def create_conversation(self, conversation_id: str) -> bool:
        """
        Create a new conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            Success status
        """
        try:
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = []
                logger.info(f"Created new conversation: {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create conversation {conversation_id}: {str(e)}")
            return False

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            Success status
        """
        try:
            if conversation_id in self._conversations:
                del self._conversations[conversation_id]
                logger.info(f"Deleted conversation: {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete conversation {conversation_id}: {str(e)}")
            return False

    def limit_history_length(self, history: List[Dict[str, Any]], max_length: int = None) -> List[Dict[str, Any]]:
        """
        Limit conversation history length.

        Args:
            history: Conversation history
            max_length: Maximum number of messages to keep

        Returns:
            Limited history
        """
        if max_length is None:
            max_length = config.chat_history_num

        # Keep the most recent messages (user + assistant pairs)
        max_messages = max_length * 2
        if len(history) > max_messages:
            limited_history = history[-max_messages:]
            logger.debug(f"Limited history from {len(history)} to {len(limited_history)} messages")
            return limited_history

        return history

    def add_message(self, conversation_id: str, message: ConversationMessage, trace_id: str) -> bool:
        """
        Add a message to conversation.

        Args:
            conversation_id: Conversation ID
            message: Message to add
            trace_id: Trace ID for logging

        Returns:
            Success status
        """
        try:
            if conversation_id not in self._conversations:
                self.create_conversation(conversation_id)

            message_dict = message.model_dump()
            self._conversations[conversation_id].append(message_dict)

            # Limit history length
            self._conversations[conversation_id] = self.limit_history_length(
                self._conversations[conversation_id]
            )

            logger.debug(f"{trace_id}: Added message to conversation {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"{trace_id}: Failed to add message to conversation {conversation_id}: {str(e)}")
            return False


# Global conversation service instance
conversation_service = ConversationService()
