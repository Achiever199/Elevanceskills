"""
app.py — Sentiment-Aware Customer Support Chatbot
Run: streamlit run app.py
"""

import nltk
import subprocess
subprocess.run(["python", "-m", "textblob.download_corpora"], capture_output=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from sentiment_engine import analyze, SentimentResult
from response_generator import build_response, generate_base_reply

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.sentiment-badge {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-left: 8px;
}
.positive  { background: #d1fae5; color: #065f46; }
.negative  { background: #fee2e2; color: #991b1b; }
.neutral   { background: #f3f4f6; color: #374151; }
.user-bubble {
    background: #3b82f6;
    color: white;
    padding: 10px 14px;
    border-radius: 18px 18px 4px 18px;
    margin: 4px 0;
    max-width: 75%;
    float: right;
    clear: both;
    font-size: 14px;
}
.bot-bubble {
    background: #f3f4f6;
    color: #1f2937;
    padding: 10px 14px;
    border-radius: 18px 18px 18px 4px;
    margin: 4px 0;
    max-width: 80%;
    float: left;
    clear: both;
    font-size: 14px;
}
.dark-mode .bot-bubble { background: #1e293b; color: #e2e8f0; }
.metric-card {
    background: #f8fafc;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # list of dicts
if "sentiment_log" not in st.session_state:
    st.session_state.sentiment_log = []     # list of SentimentResult + metadata
if "satisfaction_ratings" not in st.session_state:
    st.session_state.satisfaction_ratings = []

SENTIMENT_EMOJI = {"positive": "😊", "negative": "😟", "neutral": "😐"}
SENTIMENT_COLOR = {"positive": "#10b981", "negative": "#ef4444", "neutral": "#6b7280"}
EMOTION_EMOJI = {
    "happy": "😄", "grateful": "🙏", "frustrated": "😤", "angry": "😡",
    "sad": "😢", "anxious": "😰", "confused": "🤔", "neutral": "😐",
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Analytics Dashboard")
    st.divider()

    log = st.session_state.sentiment_log
    if log:
        total = len(log)
        pos = sum(1 for r in log if r["label"] == "positive")
        neg = sum(1 for r in log if r["label"] == "negative")
        neu = total - pos - neg
        avg_conf = sum(r["confidence"] for r in log) / total

        col1, col2 = st.columns(2)
        col1.metric("Total Messages", total)
        col2.metric("Avg Confidence", f"{avg_conf:.0%}")

        col3, col4, col5 = st.columns(3)
        col3.metric("😊 Positive", pos)
        col4.metric("😟 Negative", neg)
        col5.metric("😐 Neutral", neu)

        st.divider()
        st.subheader("Sentiment Distribution")
        fig_pie = go.Figure(go.Pie(
            labels=["Positive", "Negative", "Neutral"],
            values=[pos, neg, neu],
            hole=0.55,
            marker_colors=["#10b981", "#ef4444", "#6b7280"],
            textinfo="label+percent",
        ))
        fig_pie.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=220,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Score Over Time")
        df = pd.DataFrame(log)
        fig_line = px.line(
            df, y="score", markers=True,
            color_discrete_sequence=["#3b82f6"],
        )
        fig_line.add_hline(y=0.05,  line_dash="dot", line_color="#10b981", opacity=0.5)
        fig_line.add_hline(y=-0.05, line_dash="dot", line_color="#ef4444", opacity=0.5)
        fig_line.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=180,
            xaxis_title="Message #",
            yaxis_title="Score",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_line, use_container_width=True)

        st.subheader("Emotion Breakdown")
        emotion_counts = pd.Series([r["emotion"] for r in log]).value_counts()
        fig_bar = px.bar(
            x=emotion_counts.index,
            y=emotion_counts.values,
            color=emotion_counts.index,
            labels={"x": "Emotion", "y": "Count"},
        )
        fig_bar.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=200,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        if st.session_state.satisfaction_ratings:
            avg_sat = sum(st.session_state.satisfaction_ratings) / len(st.session_state.satisfaction_ratings)
            st.metric("⭐ Avg Satisfaction", f"{avg_sat:.1f} / 5")

        st.divider()
        if st.button("🗑️ Clear all data", use_container_width=True):
            st.session_state.messages = []
            st.session_state.sentiment_log = []
            st.session_state.satisfaction_ratings = []
            st.rerun()
    else:
        st.info("Send a message to see analytics here.")

# ── Main area ──────────────────────────────────────────────────────────────────
st.title("💬 Sentiment-Aware Customer Support")
st.caption("I detect your emotions and respond accordingly. Try expressing happiness, frustration, or confusion!")

# Chat display
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            sentiment = msg.get("sentiment")
            emoji = SENTIMENT_EMOJI.get(sentiment, "")
            badge_color = SENTIMENT_COLOR.get(sentiment, "#6b7280")
            st.markdown(
                f'<div style="text-align:right; margin-bottom:4px;">'
                f'<small style="color:{badge_color}; font-weight:600;">'
                f'{emoji} {sentiment} · score {msg.get("score", 0):+.2f} · {msg.get("emotion","")}'
                f'</small></div>'
                f'<div class="user-bubble">{msg["content"]}</div>'
                f'<div style="clear:both"></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bot-bubble">{msg["content"]}</div>'
                f'<div style="clear:both"></div>',
                unsafe_allow_html=True,
            )

        if msg.get("show_rating"):
            st.markdown("**How helpful was this response?**")
            r_col1, r_col2, r_col3, r_col4, r_col5, _ = st.columns([1,1,1,1,1,6])
            for i, col in enumerate([r_col1, r_col2, r_col3, r_col4, r_col5], 1):
                if col.button(f"{'⭐'*i}", key=f"rating_{msg['id']}_{i}"):
                    st.session_state.satisfaction_ratings.append(i)
                    msg["show_rating"] = False
                    st.rerun()

# ── Input ──────────────────────────────────────────────────────────────────────
st.divider()
with st.form("chat_form", clear_on_submit=True):
    col_in, col_btn = st.columns([5, 1])
    user_input = col_in.text_input(
        "Your message",
        placeholder="Type your message here… (e.g. 'I'm really frustrated with my order')",
        label_visibility="collapsed",
    )
    submitted = col_btn.form_submit_button("Send", use_container_width=True, type="primary")

if submitted and user_input.strip():
    # 1. Analyse sentiment
    result: SentimentResult = analyze(user_input)

    # 2. Generate reply
    base_reply = generate_base_reply(user_input)
    bot_reply = build_response(user_input, result, base_reply)

    # 3. Store messages
    msg_id = len(st.session_state.messages)
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "sentiment": result.label,
        "score": result.score,
        "emotion": result.emotion,
        "id": msg_id,
    })
    st.session_state.messages.append({
        "role": "bot",
        "content": bot_reply,
        "show_rating": True,
        "id": msg_id + 1,
    })

    # 4. Log for analytics
    st.session_state.sentiment_log.append({
        "label": result.label,
        "score": result.score,
        "confidence": result.confidence,
        "emotion": result.emotion,
        "vader": result.vader_compound,
        "textblob": result.textblob_polarity,
        "timestamp": datetime.now().isoformat(),
    })

    st.rerun()

# ── Example prompts ────────────────────────────────────────────────────────────
st.markdown("**Try these examples:**")
examples = [
    "I love this product! It works perfectly 😍",
    "This is absolutely terrible! My order still hasn't arrived!!",
    "I need help understanding how to reset my password",
    "I'm really frustrated — the app keeps crashing",
    "Thank you so much, you were incredibly helpful!",
    "I'm worried my package might be lost",
]
cols = st.columns(3)
for i, example in enumerate(examples):
    if cols[i % 3].button(example, use_container_width=True, key=f"ex_{i}"):
        result = analyze(example)
        base_reply = generate_base_reply(example)
        bot_reply = build_response(example, result, base_reply)
        msg_id = len(st.session_state.messages)
        st.session_state.messages.append({
            "role": "user", "content": example,
            "sentiment": result.label, "score": result.score,
            "emotion": result.emotion, "id": msg_id,
        })
        st.session_state.messages.append({
            "role": "bot", "content": bot_reply,
            "show_rating": True, "id": msg_id + 1,
        })
        st.session_state.sentiment_log.append({
            "label": result.label, "score": result.score,
            "confidence": result.confidence, "emotion": result.emotion,
            "vader": result.vader_compound, "textblob": result.textblob_polarity,
            "timestamp": datetime.now().isoformat(),
        })
        st.rerun()