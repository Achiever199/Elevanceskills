"""
sentiment_engine.py
Core sentiment detection using VADER + TextBlob ensemble.
Returns a SentimentResult with label, score, confidence, and emotion hints.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from dataclasses import dataclass
from typing import Literal


SentimentLabel = Literal["positive", "negative", "neutral"]

EMOTION_KEYWORDS = {
    "frustrated": ["frustrated", "frustrating", "useless", "terrible", "awful", "horrible", "worst", "broken", "failed"],
    "angry":      ["angry", "furious", "outraged", "ridiculous", "unacceptable", "disgusting", "hate"],
    "sad":        ["sad", "disappointed", "upset", "unhappy", "crying", "depressed", "heartbroken"],
    "anxious":    ["worried", "anxious", "nervous", "scared", "afraid", "concerned", "unsure"],
    "happy":      ["happy", "love", "great", "amazing", "fantastic", "wonderful", "excited", "thrilled"],
    "grateful":   ["thank", "thanks", "grateful", "appreciate", "helpful", "perfect", "excellent"],
    "confused":   ["confused", "confusing", "don't understand", "not sure", "unclear", "lost"],
}


@dataclass
class SentimentResult:
    label: SentimentLabel
    score: float          # -1.0 to 1.0
    confidence: float     # 0.0 to 1.0
    emotion: str          # fine-grained emotion hint
    vader_compound: float
    textblob_polarity: float


def detect_emotion(text: str) -> str:
    lower = text.lower()
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return emotion
    return "neutral"


def analyze(text: str) -> SentimentResult:
    """
    Ensemble: VADER (70% weight) + TextBlob (30% weight).
    VADER handles slang, punctuation intensity (!!, ALL CAPS).
    TextBlob handles grammatical polarity.
    """
    vader = SentimentIntensityAnalyzer()
    vader_scores = vader.polarity_scores(text)
    vader_compound = vader_scores["compound"]          # -1 to 1

    blob = TextBlob(text)
    tb_polarity = blob.sentiment.polarity              # -1 to 1

    # Weighted ensemble score
    ensemble_score = 0.70 * vader_compound + 0.30 * tb_polarity

    # Label — use 0.30 threshold so mild/polite queries stay neutral
    if ensemble_score >= 0.30:
        label: SentimentLabel = "positive"
    elif ensemble_score <= -0.25:
        label = "negative"
    else:
        label = "neutral"

    # Confidence: how far from zero, normalised to 0–1
    confidence = min(abs(ensemble_score) * 1.4, 1.0)

    emotion = detect_emotion(text)
    # Refine emotion when label disagrees with keyword guess
    if label == "positive" and emotion in ("frustrated", "angry", "sad", "anxious"):
        emotion = "happy"
    if label == "negative" and emotion in ("happy", "grateful"):
        emotion = "frustrated"

    return SentimentResult(
        label=label,
        score=round(ensemble_score, 4),
        confidence=round(confidence, 4),
        emotion=emotion,
        vader_compound=round(vader_compound, 4),
        textblob_polarity=round(tb_polarity, 4),
    )