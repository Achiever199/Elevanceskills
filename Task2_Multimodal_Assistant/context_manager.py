"""
context_manager.py
-------------------
Manages multi-turn conversation state, intent detection, and builds the
context block that gets folded into every prompt sent to Gemini.

Gemini's SDK supports native chat sessions, but keeping our own lightweight
ConversationContext gives us full control over intent tagging, image-memory,
and the quality-score history used by the Streamlit dashboard.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


# 9 supported intents, matched via simple keyword rules.
# Order matters: more specific intents are checked before "general".
INTENT_KEYWORDS = {
    "extract": ["extract", "read the text", "ocr", "what does it say", "transcribe"],
    "compare": ["compare", "difference between", "similar to", "versus", " vs "],
    "count": ["how many", "count", "number of"],
    "locate": ["where is", "locate", "position of", "find the"],
    "sentiment": ["mood", "feeling", "emotion", "sentiment", "tone of"],
    "identify": ["identify", "what is it", "what kind of", "what type of", "name this", "recognize"],
    "analyze": ["analyze", "analysis", "breakdown", "interpret", "assess"],
    "describe": ["describe", "what is this", "what's in", "tell me about this image", "explain the image"],
    "general": [],  # fallback
}


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    text: str
    has_image: bool = False
    image_description: Optional[str] = None
    intent: Optional[str] = None
    quality_score: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


@dataclass
class ConversationContext:
    """Holds full conversation history plus derived signals used for
    grounding new prompts and powering the quality dashboard."""

    turns: List[Turn] = field(default_factory=list)
    image_descriptions: List[str] = field(default_factory=list)
    quality_history: List[int] = field(default_factory=list)

    def detect_intent(self, text: str) -> str:
        lowered = f" {text.lower()} "
        for intent, keywords in INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in lowered:
                    return intent
        return "general"

    def add_user_turn(self, text: str, has_image: bool, intent: str) -> None:
        self.turns.append(Turn(role="user", text=text, has_image=has_image, intent=intent))

    def add_assistant_turn(
        self, text: str, quality_score: int, image_description: Optional[str] = None
    ) -> None:
        self.turns.append(Turn(role="assistant", text=text, quality_score=quality_score))
        self.quality_history.append(quality_score)
        if image_description:
            self.image_descriptions.append(image_description)

    def history_summary(self, max_turns: int = 6) -> str:
        """Condensed text summary of recent turns, injected into every new
        prompt so Gemini has conversational + visual memory even though we
        call generate_content statelessly rather than a persistent chat."""
        if not self.turns:
            return "No prior conversation."

        recent = self.turns[-max_turns:]
        lines = []
        for t in recent:
            speaker = "User" if t.role == "user" else "Assistant"
            snippet = t.text if len(t.text) <= 220 else t.text[:220] + "..."
            tag = " [image attached]" if t.has_image else ""
            lines.append(f"{speaker}{tag}: {snippet}")

        if self.image_descriptions:
            lines.append(
                "\nPreviously discussed images: " + " | ".join(self.image_descriptions[-3:])
            )
        return "\n".join(lines)

    def average_quality(self) -> float:
        if not self.quality_history:
            return 0.0
        return sum(self.quality_history) / len(self.quality_history)
