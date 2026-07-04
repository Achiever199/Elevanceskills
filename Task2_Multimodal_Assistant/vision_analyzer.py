"""
vision_analyzer.py
-------------------
Image preprocessing, prompt engineering, the Gemini Vision API call, and a
5-check response quality validator (the direct replacement for the old
Anthropic-Vision-based module).
"""

import io
import re
from typing import Dict, Optional

from PIL import Image
from google import genai

MAX_DIMENSION = 1024  # resize cap -> smaller payloads, faster + cheaper calls

SYSTEM_INSTRUCTIONS = """You are a careful, evidence-based multimodal assistant.
Rules you must always follow:
1. Ground every claim in what is actually visible in the image or stated in the text.
   Do not invent details that cannot be observed.
2. When you are not fully certain about something, say so explicitly using phrases like
   "appears to be", "it looks like", "I'm not certain, but", or "possibly".
3. Structure your answer around the user's intent (description, identification, extraction,
   comparison, analysis, counting, locating, sentiment, or general discussion).
4. If the image is ambiguous, blurry, or the question can't be fully answered from what's
   visible, state that limitation clearly instead of guessing confidently.
5. Keep responses focused - no generic filler.
"""

INTENT_INSTRUCTIONS = {
    "describe": "Give a clear, structured description covering the main subject, setting, and notable details.",
    "identify": "Identify the object/species/item as specifically as confidence allows, noting distinguishing features.",
    "extract": "Extract any visible text or data exactly as it appears, preserving structure where possible.",
    "compare": "Compare the relevant elements point by point, highlighting similarities and differences.",
    "analyze": "Provide a structured analysis: what is observed, what it suggests, and any caveats.",
    "count": "Count the requested items carefully and state the count along with what was counted.",
    "locate": "Describe the location of the requested item using its relative position in the image.",
    "sentiment": "Describe the mood/emotion conveyed, citing visual cues that support the read.",
    "general": "Respond helpfully and directly to the user's message, using the image as context if relevant.",
}


def encode_image(image: Image.Image) -> Image.Image:
    """Resize + normalize an image before sending it to the API."""
    img = image.copy()
    img.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def build_prompt(user_text: str, intent: str, history_context: str) -> str:
    intent_note = INTENT_INSTRUCTIONS.get(intent, INTENT_INSTRUCTIONS["general"])
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Detected intent: {intent}. {intent_note}\n\n"
        f"Relevant conversation context:\n{history_context}\n\n"
        f"User's current message: {user_text}\n\n"
        f"Respond now, following all rules above."
    )


def call_gemini(
    api_key: str, model_name: str, prompt: str, image: Optional[Image.Image] = None
) -> str:
    """Send prompt (+ optional image) to Gemini and return the text reply.

    Uses the current `google-genai` SDK (the old `google-generativeai`
    package is deprecated and no longer receives updates).
    """
    client = genai.Client(api_key=api_key)

    contents = [prompt]
    if image is not None:
        contents.append(image)

    response = client.models.generate_content(model=model_name, contents=contents)

    text = getattr(response, "text", None)
    return text.strip() if text else "⚠️ The model didn't return a text response (it may have been blocked by safety filters)."


# ------------------------- Response validation -------------------------
# 5 lightweight heuristic checks used to score every response 0-100%.

OBSERVATION_PATTERNS = [
    r"\bi (can |could )?see\b", r"\bthe image shows\b", r"\bvisible\b",
    r"\bappears to\b", r"\bshown in the image\b", r"\bin the image\b",
    r"\bdepicts\b", r"\bpictured\b",
]

REASONING_PATTERNS = [
    r"\bbecause\b", r"\bsince\b", r"\bsuggests?\b", r"\bindicat(e|es|ing)\b",
    r"\bthis means\b", r"\bwhich implies\b", r"\bdue to\b",
]

UNCERTAINTY_PATTERNS = [
    r"\bappears? to\b", r"\bseems?\b", r"\blikely\b", r"\bpossibly\b",
    r"\bi'?m not (fully |completely )?(certain|sure)\b", r"\bmay be\b", r"\bcould be\b",
]

OVERCONFIDENCE_PATTERNS = [
    r"\bdefinitely\b", r"\b100%\b", r"\bcertainly is\b", r"\bwithout a doubt\b",
]


def _matches_any(patterns, text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def validate_response(response_text: str, intent: str) -> Dict:
    """
    Runs 5 quality checks and returns {score, checks, word_count}.
      1. has_observations      - grounded, evidence-based language present
      2. has_reasoning         - some explanation/inference, not just facts
      3. handles_uncertainty   - uses hedging language, or isn't overconfident
      4. has_substance         - response isn't too short/thin
      5. no_hallucination_flags- avoids absolute/overconfident claims
    """
    checks = {}

    checks["has_observations"] = _matches_any(OBSERVATION_PATTERNS, response_text)
    checks["has_reasoning"] = _matches_any(REASONING_PATTERNS, response_text)

    has_uncertainty = _matches_any(UNCERTAINTY_PATTERNS, response_text)
    has_overconfidence = _matches_any(OVERCONFIDENCE_PATTERNS, response_text)
    checks["handles_uncertainty"] = has_uncertainty or not has_overconfidence

    word_count = len(response_text.split())
    checks["has_substance"] = word_count >= 15

    checks["no_hallucination_flags"] = not has_overconfidence

    passed = sum(1 for v in checks.values() if v)
    score = round((passed / len(checks)) * 100)

    return {"score": score, "checks": checks, "word_count": word_count}
