"""
language_detector.py
Detects language of input text using langdetect.
Handles mixed-language inputs by splitting and detecting per segment.
"""

from langdetect import detect, detect_langs, LangDetectException
from dataclasses import dataclass
from typing import List

SUPPORTED_LANGUAGES = {
    "en":    {"name": "English",    "flag": "🇬🇧", "native": "English"},
    "hi":    {"name": "Hindi",      "flag": "🇮🇳", "native": "हिंदी"},
    "fr":    {"name": "French",     "flag": "🇫🇷", "native": "Français"},
    "es":    {"name": "Spanish",    "flag": "🇪🇸", "native": "Español"},
    "de":    {"name": "German",     "flag": "🇩🇪", "native": "Deutsch"},
    "zh-cn": {"name": "Chinese",    "flag": "🇨🇳", "native": "中文"},
    "ar":    {"name": "Arabic",     "flag": "🇸🇦", "native": "العربية"},
    "pt":    {"name": "Portuguese", "flag": "🇧🇷", "native": "Português"},
    "ja":    {"name": "Japanese",   "flag": "🇯🇵", "native": "日本語"},
    "ru":    {"name": "Russian",    "flag": "🇷🇺", "native": "Русский"},
}

GREETINGS = {
    "en":    "Hello! How can I help you?",
    "hi":    "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?",
    "fr":    "Bonjour! Comment puis-je vous aider?",
    "es":    "¡Hola! ¿Cómo puedo ayudarte?",
    "de":    "Hallo! Wie kann ich Ihnen helfen?",
    "zh-cn": "你好！我能帮助你吗？",
    "ar":    "مرحباً! كيف يمكنني مساعدتك؟",
    "pt":    "Olá! Como posso ajudar você?",
    "ja":    "こんにちは！どのようにお手伝いできますか？",
    "ru":    "Привет! Чем я могу помочь?",
}

SWITCH_ACK = {
    "en":    "I noticed you switched to English. I'll continue in English.",
    "hi":    "मैंने देखा कि आपने हिंदी में स्विच किया। मैं हिंदी में जारी रखूँगा।",
    "fr":    "J'ai remarqué que vous avez changé en français. Je continuerai en français.",
    "es":    "Noté que cambiaste al español. Continuaré en español.",
    "de":    "Ich habe bemerkt, dass Sie zu Deutsch gewechselt haben. Ich werde auf Deutsch fortfahren.",
    "zh-cn": "我注意到您切换到了中文。我将继续用中文。",
    "ar":    "لاحظت أنك تحولت إلى العربية. سأكمل باللغة العربية.",
    "pt":    "Notei que você mudou para português. Continuarei em português.",
    "ja":    "日本語に切り替えたことに気づきました。日本語で続けます。",
    "ru":    "Я заметил, что вы переключились на русский. Продолжу на русском.",
}

# Short common English words/phrases that langdetect often misidentifies
ENGLISH_FORCE = {
    "hello", "hi", "hey", "yes", "no", "ok", "okay", "bye", "thanks",
    "help", "good", "bad", "what", "why", "how", "where", "when", "who",
    "please", "sorry", "great", "nice", "cool", "sure", "welcome", "test",
    "tell me", "i am", "i need", "i want", "can you", "what is", "who is",
    "how are you", "thank you", "good morning", "good night", "good evening",
}


@dataclass
class DetectionResult:
    lang_code: str
    confidence: float
    lang_name: str
    flag: str
    is_supported: bool
    all_candidates: List[dict]


def detect_language(text: str) -> DetectionResult:
    """Detect language with confidence scores."""
    if not text or len(text.strip()) < 2:
        return DetectionResult("en", 1.0, "English", "🇬🇧", True, [])

    lower = text.strip().lower()

    # Force English for known short common words/phrases
    if lower in ENGLISH_FORCE:
        return DetectionResult("en", 0.99, "English", "🇬🇧", True, [])

    # If all ASCII and short (<= 20 chars), likely English
    if len(lower) <= 20 and all(ord(c) < 128 for c in lower):
        return DetectionResult("en", 0.85, "English", "🇬🇧", True, [])

    try:
        candidates = detect_langs(text)
        top = candidates[0]
        lang_code = top.lang
        confidence = round(top.prob, 3)

        # Normalize Chinese variants
        if lang_code in ("zh-tw", "zh"):
            lang_code = "zh-cn"

        # Low confidence + pure ASCII → default English
        if confidence < 0.55 and all(ord(c) < 128 for c in text):
            lang_code = "en"
            confidence = 0.65

        info = SUPPORTED_LANGUAGES.get(lang_code, {
            "name": lang_code.upper(),
            "flag": "🌐",
            "native": lang_code,
        })

        all_cands = [
            {"lang": c.lang, "prob": round(c.prob, 3)}
            for c in candidates[:3]
        ]

        return DetectionResult(
            lang_code=lang_code,
            confidence=confidence,
            lang_name=info["name"],
            flag=info["flag"],
            is_supported=lang_code in SUPPORTED_LANGUAGES,
            all_candidates=all_cands,
        )

    except LangDetectException:
        return DetectionResult("en", 0.5, "English", "🇬🇧", True, [])


def is_mixed_language(text: str) -> bool:
    """Check if text contains multiple languages."""
    words = text.split()
    if len(words) < 4:
        return False
    first_half = " ".join(words[:len(words)//2])
    second_half = " ".join(words[len(words)//2:])
    try:
        lang1 = detect(first_half)
        lang2 = detect(second_half)
        return lang1 != lang2
    except Exception:
        return False