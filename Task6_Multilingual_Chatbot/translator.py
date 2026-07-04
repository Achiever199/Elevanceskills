"""
translator.py
Handles translation using deep-translator (Google Translate backend, free).
No API key required.
"""

from deep_translator import GoogleTranslator
from typing import Optional

# Map our lang codes to deep-translator codes
LANG_MAP = {
    "en":    "en",
    "hi":    "hi",
    "fr":    "fr",
    "es":    "es",
    "de":    "de",
    "zh-cn": "zh-CN",
    "ar":    "ar",
    "pt":    "pt",
    "ja":    "ja",
    "ru":    "ru",
}


def translate(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text from source_lang to target_lang.
    Returns original text if translation fails or langs are the same.
    """
    if not text or not text.strip():
        return text

    src = LANG_MAP.get(source_lang, source_lang)
    tgt = LANG_MAP.get(target_lang, target_lang)

    # Normalize source lang
    if src in ("zh-tw", "zh"):
        src = "zh-CN"

    if src == tgt:
        return text

    try:
        translated = GoogleTranslator(source=src, target=tgt).translate(text)
        return translated if translated else text
    except Exception:
        return text  # Fallback to original on any error


def translate_to_english(text: str, source_lang: str) -> str:
    """Translate any language to English for intent processing."""
    return translate(text, source_lang, "en")


def translate_from_english(text: str, target_lang: str) -> str:
    """Translate English response to target language."""
    return translate(text, "en", target_lang)
