"""
response_generator.py
Maps SentimentResult → a contextually appropriate response prefix + tone.
"""

import random
from sentiment_engine import SentimentResult

# ── Response templates keyed by (label, emotion) ──────────────────────────────
TEMPLATES: dict[str, dict[str, list[str]]] = {
    "positive": {
        "happy": [
            "That's wonderful to hear! 😊 ",
            "I'm so glad you're feeling great! ",
            "Awesome — love the enthusiasm! 🎉 ",
        ],
        "grateful": [
            "Thank you so much for the kind words! 🙏 ",
            "It's our pleasure — really glad we could help! ",
            "That means a lot to us. Thank you! 😊 ",
        ],
        "default": [
            "Great to hear that! ",
            "Glad things are going well! 😊 ",
            "That's positive news! ",
        ],
    },
    "negative": {
        "frustrated": [
            "I'm really sorry you're dealing with this — that sounds frustrating. Let me help you sort it out. ",
            "That must be annoying, and I completely understand. I'll do my best to fix this for you. ",
            "I hear you, and I'm sorry for the trouble. Let's resolve this right away. ",
        ],
        "angry": [
            "I sincerely apologise — you deserve better than this. Let me escalate this immediately. ",
            "I completely understand your frustration, and I'm sorry we've let you down. I'm on it. ",
            "That's unacceptable, and I take full responsibility. Let me make this right for you. ",
        ],
        "sad": [
            "I'm really sorry to hear you're feeling this way. You're not alone — I'm here to help. ",
            "That sounds really tough. I'm sorry, and I want to make things better for you. ",
            "I understand, and I'm sorry for the experience. Let's work through this together. ",
        ],
        "anxious": [
            "I understand this feels uncertain — please don't worry, I'll walk you through every step. ",
            "It's completely okay to have questions. I'm here to make this as clear as possible for you. ",
            "Take a breath — we'll sort this out together, one step at a time. ",
        ],
        "default": [
            "I'm sorry to hear that. Let me help you with this right away. ",
            "I apologise for any inconvenience. I'm here to help resolve this. ",
            "Thank you for letting me know — I'll do my best to assist you. ",
        ],
    },
    "neutral": {
        "confused": [
            "Happy to clarify! Let me break this down simply. ",
            "Good question — let me explain this step by step. ",
            "Sure, let me make this clearer for you. ",
        ],
        "default": [
            "Thanks for reaching out! ",
            "Of course — happy to help with that. ",
            "Sure thing! Here's what I can tell you: ",
        ],
    },
}

# Escalation message for high-confidence negative
ESCALATION_NOTE = (
    "\n\n---\n*Would you like me to connect you with a senior support agent? "
    "Just say **'escalate'** and I'll transfer you right away.*"
)


def get_response_prefix(result: SentimentResult) -> str:
    label_templates = TEMPLATES.get(result.label, TEMPLATES["neutral"])
    emotion_templates = label_templates.get(result.emotion, label_templates.get("default", [""]))
    return random.choice(emotion_templates)


def build_response(user_message: str, result: SentimentResult, base_reply: str) -> str:
    """
    Combines sentiment-aware prefix + the actual reply content.
    Appends escalation offer for high-confidence negative messages.
    """
    prefix = get_response_prefix(result)
    response = prefix + base_reply

    if result.label == "negative" and result.confidence > 0.6:
        response += ESCALATION_NOTE

    return response


# ── Simple rule-based reply logic (replace with LLM call in production) ────────
FAQ_RESPONSES: dict[str, str] = {
    "refund":    "To process a refund, please share your order number and I'll initiate it within 24 hours.",
    "shipping":  "Standard shipping takes 3–5 business days. Express shipping is 1–2 days.",
    "cancel":    "I can cancel your order if it hasn't been dispatched yet. Please share your order ID.",
    "password":  "To reset your password, click 'Forgot Password' on the login page and follow the steps.",
    "account":   "I can help you update your account details. What would you like to change?",
    "price":     "You can check our latest pricing on our website. Do you want me to share the link?",
    "discount":  "We currently have a 10% welcome discount — use code WELCOME10 at checkout.",
    "contact":   "You can reach our team at support@example.com or call 1-800-EXAMPLE.",
    "hours":     "Our support team is available Monday–Friday, 9 AM to 6 PM (IST).",
    "broken":    "I'm sorry to hear about the issue! Please describe the problem and I'll troubleshoot it with you.",
    "late":      "I apologise for the delay! Let me check the status of your order right now.",
}

DEFAULT_REPLIES = [
    "I'm here to help! Could you share more details so I can assist you better?",
    "Thanks for your message. Let me look into that for you.",
    "I'd be happy to assist with that. Could you give me a bit more context?",
]


def generate_base_reply(user_message: str) -> str:
    lower = user_message.lower()
    for keyword, reply in FAQ_RESPONSES.items():
        if keyword in lower:
            return reply
    return random.choice(DEFAULT_REPLIES)