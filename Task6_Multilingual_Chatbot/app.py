"""
app.py — Multilingual Chatbot
Supports: English, Hindi, French, Spanish, German,
          Chinese, Arabic, Portuguese, Japanese, Russian
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
from datetime import datetime

from language_detector import (
    detect_language, is_mixed_language,
    SUPPORTED_LANGUAGES, GREETINGS, SWITCH_ACK, DetectionResult
)
from translator import translate_to_english, translate_from_english
from intent_engine import (
    detect_intent, build_context_note, ConversationContext, INTENTS
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multilingual Chatbot",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.user-bubble {
    background: #6366f1; color: white;
    padding: 10px 16px; border-radius: 18px 18px 4px 18px;
    margin: 6px 0; max-width: 75%; float: right; clear: both;
    font-size: 14px; line-height: 1.5;
}
.bot-bubble {
    background: #f0f9ff; color: #1a1a1a; border: 1px solid #bae6fd;
    padding: 12px 16px; border-radius: 18px 18px 18px 4px;
    margin: 6px 0; max-width: 82%; float: left; clear: both;
    font-size: 14px; line-height: 1.6;
}
.lang-badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 12px; font-size: 11px; font-weight: 600;
    background: #e0e7ff; color: #3730a3; margin-left: 6px;
}
.switch-banner {
    background: #fef3c7; border: 1px solid #f59e0b;
    border-radius: 8px; padding: 6px 12px;
    font-size: 12px; color: #92400e;
    margin: 4px 0; clear: both;
}
.stat-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 12px; text-align: center;
}
.context-box {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 8px; padding: 10px; font-size: 12px;
    margin: 6px 0; clear: both;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = ConversationContext()
if "lang_history" not in st.session_state:
    st.session_state.lang_history = []

ctx: ConversationContext = st.session_state.context

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌍 Multilingual Chatbot")
    st.caption("10 Languages · Auto Detection · Context Retention")
    st.divider()

    # Supported languages list
    st.subheader("🗣️ Supported Languages")
    for code, info in SUPPORTED_LANGUAGES.items():
        st.markdown(
            f"{info['flag']} **{info['name']}** — *{info['native']}*"
        )

    st.divider()

    # Live session stats
    st.subheader("📊 Session Analytics")
    if st.session_state.lang_history:
        total = len(st.session_state.lang_history)
        switches = ctx.language_switches
        langs_used = list(set(st.session_state.lang_history))

        col1, col2 = st.columns(2)
        col1.metric("Messages", total)
        col2.metric("Lang Switches", switches)

        st.metric("Languages Used", len(langs_used))

        # Language usage pie chart
        lang_counts = pd.Series(st.session_state.lang_history).value_counts()
        lang_labels = [
            f"{SUPPORTED_LANGUAGES.get(c, {}).get('flag','🌐')} {SUPPORTED_LANGUAGES.get(c, {}).get('name', c)}"
            for c in lang_counts.index
        ]
        fig = go.Figure(go.Pie(
            labels=lang_labels,
            values=lang_counts.values,
            hole=0.5,
            textinfo="label+percent",
        ))
        fig.update_layout(
            height=220,
            margin=dict(t=10, b=10, l=0, r=0),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Language switch timeline
        if len(st.session_state.lang_history) > 1:
            st.subheader("🔄 Language Timeline")
            timeline_df = pd.DataFrame({
                "Turn": range(1, len(st.session_state.lang_history) + 1),
                "Language": [
                    SUPPORTED_LANGUAGES.get(l, {}).get("name", l)
                    for l in st.session_state.lang_history
                ]
            })
            fig2 = px.scatter(
                timeline_df, x="Turn", y="Language",
                color="Language",
                size=[10] * len(timeline_df),
            )
            fig2.update_layout(
                height=180,
                margin=dict(t=10, b=10, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Send a message to see analytics.")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.context = ConversationContext()
        st.session_state.lang_history = []
        st.rerun()

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("🌍 Multilingual Conversational Chatbot")
st.caption("Type in any of 10 languages — I auto-detect and respond in your language!")

# ── Chat display ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        lang_info = SUPPORTED_LANGUAGES.get(msg.get("lang", "en"), {})
        flag = lang_info.get("flag", "🌐")
        lang_name = lang_info.get("name", msg.get("lang", ""))
        confidence = msg.get("confidence", 1.0)
        st.markdown(
            f'<div style="text-align:right;margin-bottom:2px;">'
            f'<small style="color:#6366f1;font-weight:600">'
            f'{flag} {lang_name} · {confidence:.0%} confidence</small></div>'
            f'<div class="user-bubble">{msg["content"]}</div>'
            f'<div style="clear:both"></div>',
            unsafe_allow_html=True,
        )

    elif msg["role"] == "switch_banner":
        st.markdown(
            f'<div class="switch-banner">🔄 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )

    elif msg["role"] == "bot":
        lang_info = SUPPORTED_LANGUAGES.get(msg.get("lang", "en"), {})
        flag = lang_info.get("flag", "🌐")
        # Render markdown in bot reply
        st.markdown(
            f'<div style="margin-bottom:2px;">'
            f'<small style="color:#0284c7;font-weight:600">'
            f'🤖 Assistant {flag}</small></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="bot-bubble">{msg["content"]}</div>'
            f'<div style="clear:both"></div>',
            unsafe_allow_html=True,
        )

        # Show context retention info
        if msg.get("show_context") and ctx.turn_count > 2:
            topics = ", ".join(ctx.topics_discussed[-3:]) if ctx.topics_discussed else "general"
            langs = ", ".join([
                SUPPORTED_LANGUAGES.get(l, {}).get("name", l)
                for l in list(set(st.session_state.lang_history))[-3:]
            ])
            st.markdown(
                f'<div class="context-box">🧠 <strong>Context retained</strong> · '
                f'Turn {ctx.turn_count} · Languages used: {langs} · '
                f'Topics: {topics}</div>',
                unsafe_allow_html=True,
            )

# ── Input ──────────────────────────────────────────────────────────────────────
st.divider()
with st.form("chat_form", clear_on_submit=True):
    col_in, col_btn = st.columns([5, 1])
    user_input = col_in.text_input(
        "Message",
        placeholder="Type in any language… (e.g. Bonjour! / नमस्ते / Hola!)",
        label_visibility="collapsed",
    )
    submitted = col_btn.form_submit_button("Send", type="primary", use_container_width=True)


def process_message(text: str):
    """Full pipeline: detect → translate → intent → respond → translate back."""

    # 1. Detect language
    detection: DetectionResult = detect_language(text)
    detected_lang = detection.lang_code
    mixed = is_mixed_language(text)

    # 2. Check for language switch
    switch_note = build_context_note(ctx, detected_lang)

    # 3. Translate to English for intent processing
    if detected_lang != "en":
        text_en = translate_to_english(text, detected_lang)
    else:
        text_en = text

    # 4. Detect intent
    intent_result = detect_intent(text_en)

    # 5. Build English response
    response_en = intent_result.response_en

    # Add context-aware additions
    if ctx.turn_count > 0 and detected_lang != ctx.last_lang:
        # Language switched — acknowledge it
        ctx.language_switches += 1

    if "{lang}" in response_en:
        response_en = response_en.replace(
            "{lang}", SUPPORTED_LANGUAGES.get(detected_lang, {}).get("name", detected_lang)
        )

    # 6. Translate response back to user's language
    if detected_lang != "en":
        response_final = translate_from_english(response_en, detected_lang)
        # If translation fails (returns English), use greeting in that lang
        if response_final == response_en and detected_lang in GREETINGS:
            response_final = response_en  # keep English as fallback
    else:
        response_final = response_en

    # 7. Update context
    ctx.turn_count += 1
    ctx.last_intent = intent_result.intent
    ctx.topics_discussed.append(intent_result.intent)
    if len(ctx.topics_discussed) > 10:
        ctx.topics_discussed = ctx.topics_discussed[-10:]

    prev_lang = ctx.last_lang
    ctx.last_lang = detected_lang
    if detected_lang not in ctx.languages_used:
        ctx.languages_used.append(detected_lang)

    st.session_state.lang_history.append(detected_lang)

    # 8. Store messages
    st.session_state.messages.append({
        "role": "user",
        "content": text,
        "lang": detected_lang,
        "confidence": detection.confidence,
    })

    # Show language switch banner
    if switch_note and prev_lang != detected_lang:
        switch_msg = SWITCH_ACK.get(
            detected_lang,
            f"Language switched to {SUPPORTED_LANGUAGES.get(detected_lang, {}).get('name', detected_lang)}."
        )
        st.session_state.messages.append({
            "role": "switch_banner",
            "content": switch_msg,
        })

    st.session_state.messages.append({
        "role": "bot",
        "content": response_final,
        "lang": detected_lang,
        "intent": intent_result.intent,
        "show_context": ctx.turn_count > 2,
    })


if submitted and user_input.strip():
    process_message(user_input.strip())
    st.rerun()

# ── Example prompts in multiple languages ──────────────────────────────────────
st.markdown("**🌐 Try these in different languages:**")
examples = [
    ("🇬🇧", "Hello! What can you do?"),
    ("🇮🇳", "नमस्ते! आप कौन सी भाषाएँ बोलते हैं?"),
    ("🇫🇷", "Bonjour! Comment ça va?"),
    ("🇪🇸", "¡Hola! ¿Cómo estás?"),
    ("🇩🇪", "Hallo! Wie geht es Ihnen?"),
    ("🇨🇳", "你好！你会说什么语言？"),
    ("🇸🇦", "مرحبا! كيف حالك؟"),
    ("🇧🇷", "Olá! Como você está?"),
    ("🇯🇵", "こんにちは！お元気ですか？"),
    ("🇷🇺", "Привет! Как дела?"),
    ("🇬🇧", "Tell me a joke!"),
    ("🇮🇳", "मेरी मदद करो"),
]
cols = st.columns(4)
for i, (flag, example) in enumerate(examples):
    if cols[i % 4].button(f"{flag} {example[:25]}", key=f"ex_{i}", use_container_width=True):
        process_message(example)
        st.rerun()
