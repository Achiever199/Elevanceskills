"""
intent_engine.py
Processes user queries (in English after translation) and generates responses.
Maintains conversation context across language switches.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional

INTENTS = {
    "greeting": {
        "keywords": ["hello", "hi", "hey", "good morning", "good evening",
                     "good afternoon", "howdy", "greetings", "what's up", "sup"],
        "responses": [
            "Hello! I'm your multilingual assistant. I can understand and respond in English, Hindi, French, Spanish, German, Chinese, Arabic, Portuguese, Japanese, and Russian. How can I help you today?",
            "Hi there! I speak 10 languages fluently. Feel free to switch languages anytime — I'll follow along!",
        ],
    },
    "farewell": {
        "keywords": ["bye", "goodbye", "see you", "farewell", "take care",
                     "good night", "later", "ciao", "adios"],
        "responses": [
            "Goodbye! It was a pleasure chatting with you. Feel free to return anytime!",
            "Take care! Remember, you can always come back and chat in any language you prefer.",
        ],
    },
    "language_question": {
        "keywords": ["what language", "which language", "how many language",
                     "language do you speak", "language can you", "what languages"],
        "responses": [
            "I can understand and respond in 10 languages: English 🇬🇧, Hindi 🇮🇳, French 🇫🇷, Spanish 🇪🇸, German 🇩🇪, Chinese 🇨🇳, Arabic 🇸🇦, Portuguese 🇧🇷, Japanese 🇯🇵, and Russian 🇷🇺. Just type in any of these languages!",
        ],
    },
    "name": {
        "keywords": ["your name", "who are you", "what are you", "call you",
                     "are you a bot", "are you ai", "are you human"],
        "responses": [
            "I'm a multilingual AI assistant built for an internship project. I use language detection and translation to understand you in any of 10 supported languages and respond in kind!",
        ],
    },
    "help": {
        "keywords": ["help", "how does this work", "what can you do",
                     "features", "capabilities", "instructions", "guide"],
        "responses": [
            "Here's what I can do:\n\n"
            "🌍 **Auto Language Detection** — I detect your language automatically\n"
            "🔄 **Language Switching** — Switch languages mid-conversation, I'll follow\n"
            "🧠 **Context Retention** — I remember our conversation history\n"
            "🔀 **Mixed Language** — Handle mixed language inputs\n"
            "💬 **10 Languages** — EN, HI, FR, ES, DE, ZH, AR, PT, JA, RU\n\n"
            "Just type your message in any supported language to get started!",
        ],
    },
    "india": {
        "keywords": ["what is india", "india", "tell me about india",
                     "indian", "india country"],
        "responses": [
            "India 🇮🇳 is the world's largest democracy and the seventh-largest country by area. It is home to over 1.4 billion people, 22 official languages, and a rich cultural heritage spanning thousands of years. India is known for its diversity in language, religion, cuisine, and geography — from the Himalayas in the north to tropical coastlines in the south.",
            "India is a South Asian country known for its ancient civilization, diverse culture, and rapid economic growth. It is the birthplace of major religions like Hinduism, Buddhism, Jainism, and Sikhism. New Delhi is its capital, and Mumbai is its financial hub.",
        ],
    },
    "geography": {
        "keywords": ["what is", "tell me about", "explain", "describe",
                     "country", "city", "capital", "continent", "ocean", "river", "mountain"],
        "responses": [
            "That's an interesting geography question! I can provide general information about world geography. Could you be more specific about what you'd like to know?",
            "I'd be happy to discuss world geography! Please share more details about the specific place or topic you're curious about.",
        ],
    },
    "weather": {
        "keywords": ["weather", "temperature", "forecast", "rain", "sunny",
                     "cold", "hot", "climate", "snow", "storm"],
        "responses": [
            "I don't have access to real-time weather data, but weather apps like Google Weather or AccuWeather can give you live forecasts for any city. Would you like to know something else?",
        ],
    },
    "time": {
        "keywords": ["time", "what time", "current time", "date", "today",
                     "what day", "what year"],
        "responses": [
            "I don't have access to real-time clock data. You can check your device's clock for the current time and date. Is there something else I can help with?",
        ],
    },
    "thanks": {
        "keywords": ["thank you", "thanks", "thank", "appreciate", "grateful",
                     "awesome", "great", "excellent", "wonderful", "perfect"],
        "responses": [
            "You're welcome! I'm happy to help. Feel free to ask me anything else — in any language!",
            "Glad I could help! Don't hesitate to ask more questions.",
        ],
    },
    "feeling": {
        "keywords": ["how are you", "how do you do", "are you okay",
                     "how are things", "how's it going"],
        "responses": [
            "I'm doing great, thank you for asking! As an AI, I'm always ready to assist you. How are you doing today?",
        ],
    },
    "joke": {
        "keywords": ["joke", "funny", "humor", "laugh", "tell me a joke"],
        "responses": [
            "Why do programmers prefer dark mode? Because light attracts bugs! 😄",
            "Why did the multilingual bot cross the road? To get to the other language! 🌍",
        ],
    },
    "translate_request": {
        "keywords": ["translate this", "how do you say", "what does mean",
                     "say in french", "say in hindi", "say in spanish", "translate to"],
        "responses": [
            "I can help with translation! Just type your text in any of my 10 supported languages and I'll detect it and respond in kind. For direct translation requests, mention the target language (e.g. 'Say hello in French').",
        ],
    },
    "fallback": {
        "keywords": [],
        "responses": [
            "I received your message and detected your language! I'm a multilingual assistant — feel free to ask me about languages, translation, general knowledge, or just have a conversation in any of my 10 supported languages.",
            "Interesting! I understood your message. I can converse in 10 languages and help with general questions. Could you elaborate a bit more so I can assist you better?",
            "Got your message in {lang}! I'm here to help with multilingual conversations. Feel free to ask me anything or switch to another language anytime!",
        ],
    },
}


@dataclass
class ConversationContext:
    """Tracks conversation state across language switches."""
    turn_count: int = 0
    languages_used: List[str] = field(default_factory=list)
    last_lang: str = "en"
    last_intent: str = ""
    topics_discussed: List[str] = field(default_factory=list)
    language_switches: int = 0


@dataclass
class IntentResult:
    intent: str
    response_en: str
    confidence: float


def detect_intent(text_en: str) -> IntentResult:
    """Match text (in English) to best intent using keyword scoring."""
    lower = text_en.lower()

    best_intent = "fallback"
    best_score = 0

    for intent_name, intent_data in INTENTS.items():
        if intent_name == "fallback":
            continue
        score = sum(1 for kw in intent_data["keywords"] if kw in lower)
        if score > best_score:
            best_score = score
            best_intent = intent_name

    confidence = min(best_score / 2, 1.0) if best_score > 0 else 0.3
    responses = INTENTS[best_intent]["responses"]
    response = random.choice(responses)

    return IntentResult(
        intent=best_intent,
        response_en=response,
        confidence=confidence,
    )


def build_context_note(ctx: ConversationContext, new_lang: str) -> Optional[str]:
    if ctx.last_lang and ctx.last_lang != new_lang and ctx.turn_count > 0:
        return f"[LANGUAGE_SWITCH:{new_lang}]"
    return None