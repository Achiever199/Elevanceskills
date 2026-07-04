"""
conversation.py
----------------
Multi-turn conversation state for the chat tab, plus follow-up-question
handling: short/ambiguous follow-ups ("what about efficiency?", "explain
that more") are expanded with recent topic context before being sent to
the retriever, so RAG search doesn't go in blind on a pronoun-only query.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

FOLLOWUP_MARKERS = [
    r"\bthat\b", r"\bthis\b", r"\bit\b", r"\bthose\b", r"\bthese\b",
    r"^what about\b", r"^and\b", r"^also\b", r"\bmore\b", r"\bwhy\b$",
    r"^explain (more|further)\b", r"^how so\b", r"^can you elaborate\b",
]


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    text: str
    sources: List[str] = field(default_factory=list)  # paper ids used as evidence


@dataclass
class ConversationContext:
    turns: List[Turn] = field(default_factory=list)
    last_topic: Optional[str] = None  # last substantive (non-follow-up) query

    def is_followup(self, query: str) -> bool:
        q = query.strip().lower()
        if len(q.split()) <= 4:
            return True
        return any(re.search(p, q) for p in FOLLOWUP_MARKERS)

    def expand_query(self, query: str) -> str:
        """If the query looks like a follow-up, prepend the last substantive
        topic so retrieval has something concrete to search for."""
        if self.is_followup(query) and self.last_topic:
            return f"{self.last_topic}. {query}"
        return query

    def add_user_turn(self, text: str) -> None:
        self.turns.append(Turn(role="user", text=text))
        if not self.is_followup(text):
            self.last_topic = text

    def add_assistant_turn(self, text: str, sources: Optional[List[str]] = None) -> None:
        self.turns.append(Turn(role="assistant", text=text, sources=sources or []))

    def history_text(self, max_turns: int = 4) -> str:
        if not self.turns:
            return ""
        recent = self.turns[-max_turns:]
        lines = []
        for t in recent:
            speaker = "User" if t.role == "user" else "Assistant"
            snippet = t.text if len(t.text) <= 300 else t.text[:300] + "..."
            lines.append(f"{speaker}: {snippet}")
        return "\n".join(lines)

    def reset(self) -> None:
        self.turns = []
        self.last_topic = None
