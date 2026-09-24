# core/memory.py - Context Window & Conversation History Subsystem
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

@dataclass
class ChatMessage:
    """Data class representing a structured message in the conversation history."""
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Export clean dictionary suitable for Groq API payload."""
        data = {"role": self.role}
        if self.content is not None:
            data["content"] = self.content
        if self.name:
            data["name"] = self.name
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        return data

    @property
    def token_estimate(self) -> int:
        """Heuristic token estimation (~1.3 tokens per word + metadata)."""
        text = self.content or ""
        if self.tool_calls:
            text += json.dumps(self.tool_calls)
        words = len(text.split())
        return max(4, int(words * 1.3) + 3)


class MemoryManager:
    """
    Manages chat history, sliding window pruning, token budgeting,
    and Streamlit session state serialization.
    """
    def __init__(
        self,
        system_prompt: str = "You are a helpful AI Agent equipped with tools.",
        max_token_budget: int = 2000
    ):
        self.system_prompt = system_prompt
        self.max_token_budget = max_token_budget
        self.history: List[ChatMessage] = []
        
        # System message is pinned at index 0
        self.history.append(ChatMessage(role="system", content=self.system_prompt))

    @property
    def messages(self) -> List[ChatMessage]:
        """Exposes conversation history to external modules like Streamlit app.py."""
        return self.history
    
    def add_message(self, role: str, content: Optional[str] = None, **kwargs) -> ChatMessage:
        """Appends a new ChatMessage instance to memory history."""
        msg = ChatMessage(role=role, content=content, **kwargs)
        self.history.append(msg)
        return msg

    def get_total_tokens(self) -> int:
        """Calculates total estimated tokens across full history."""
        return sum(msg.token_estimate for msg in self.history)

    def get_pruned_messages(self) -> List[Dict[str, Any]]:
        """
        Returns a pruned list of message dictionaries within max_token_budget.
        Guarantees system message (index 0) is NEVER dropped.
        """
        if not self.history:
            return []

        system_msg = self.history[0]
        conv_msgs = self.history[1:]

        current_tokens = system_msg.token_estimate
        selected_msgs: List[ChatMessage] = []

        # Iterate backwards (FIFO sliding window from most recent)
        for msg in reversed(conv_msgs):
            msg_tokens = msg.token_estimate
            if current_tokens + msg_tokens <= self.max_token_budget:
                selected_msgs.insert(0, msg)
                current_tokens += msg_tokens
            else:
                # Budget limit reached, drop older messages
                break

        pruned_list = [system_msg.to_dict()] + [m.to_dict() for m in selected_msgs]
        return pruned_list

    def serialize(self) -> List[Dict[str, Any]]:
        """Serializes memory history for Streamlit st.session_state storage."""
        return [asdict(msg) for msg in self.history]

    @classmethod
    def deserialize(cls, data: List[Dict[str, Any]], system_prompt: str) -> "MemoryManager":
        """Reconstructs MemoryManager instance from session state data."""
        mgr = cls(system_prompt=system_prompt)
        mgr.history = [ChatMessage(**item) for item in data]
        return mgr
