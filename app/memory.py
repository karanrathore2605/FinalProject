import logging
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.config import MAX_HISTORY_MESSAGES

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a single message in a conversation."""
    role: str  # "user", "assistant", or "system"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class ConversationManager:
    """
    Manages conversational memory and chat history per session.
    Provides session isolation, configurable history limits, and thread-safe operations.
    """

    def __init__(self, max_history: int = MAX_HISTORY_MESSAGES):
        self.max_history = max_history
        self._sessions: Dict[str, List[Dict[str, str]]] = {}

    def create_session(self) -> str:
        """Create a new unique session ID."""
        session_id = f"session_{uuid.uuid4().hex[:10]}"
        self._sessions[session_id] = []
        return session_id

    def get_history(
        self,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Retrieve recent conversation history for a given session.
        Returns a list of message dicts: [{'role': 'user'|'assistant', 'content': str}].
        """
        if not session_id or session_id not in self._sessions:
            return []

        history = self._sessions[session_id]
        hist_limit = limit if limit is not None else self.max_history
        if hist_limit > 0 and len(history) > hist_limit:
            return [dict(m) for m in history[-hist_limit:]]
        return [dict(m) for m in history]

    def add_user_message(self, session_id: str, content: str) -> None:
        """Add a user message to session history."""
        if not session_id or not content.strip():
            return
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": "user", "content": content.strip()})
        self._enforce_limit(session_id)

    def add_assistant_message(self, session_id: str, content: str) -> None:
        """Add an assistant message to session history."""
        if not session_id or not content.strip():
            return
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": "assistant", "content": content.strip()})
        self._enforce_limit(session_id)

    def add_turn(self, session_id: str, user_query: str, assistant_response: str) -> None:
        """Record a completed conversation turn (user query + assistant response)."""
        if not session_id:
            return
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        if user_query.strip():
            self._sessions[session_id].append({"role": "user", "content": user_query.strip()})
        if assistant_response.strip():
            self._sessions[session_id].append({"role": "assistant", "content": assistant_response.strip()})
        self._enforce_limit(session_id)

    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a specific session."""
        if session_id in self._sessions:
            self._sessions[session_id] = []
            return True
        return False

    def delete_session(self, session_id: str) -> bool:
        """Completely remove a session and its history."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self._sessions

    def get_session_count(self) -> int:
        """Return the number of active sessions."""
        return len(self._sessions)

    def get_session_message_count(self, session_id: str) -> int:
        """Return total messages stored for a session."""
        return len(self._sessions.get(session_id, []))

    def _enforce_limit(self, session_id: str) -> None:
        """Trim history to prevent unbounded memory growth."""
        if self.max_history > 0 and session_id in self._sessions:
            buffer_limit = max(self.max_history * 2, 20)
            if len(self._sessions[session_id]) > buffer_limit:
                self._sessions[session_id] = self._sessions[session_id][-buffer_limit:]


# Global singleton instance
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """Retrieve or initialize the global ConversationManager singleton."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager(max_history=MAX_HISTORY_MESSAGES)
    return _conversation_manager


def reset_conversation_manager() -> None:
    """Reset the global conversation manager (useful for testing)."""
    global _conversation_manager
    _conversation_manager = None
