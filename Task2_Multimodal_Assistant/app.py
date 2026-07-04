"""
app.py
------
Streamlit UI for the multimodal (text + image) AI assistant.
Powered by Google Gemini's free-tier API - no Anthropic key required.
"""

import streamlit as st
from PIL import Image
import plotly.graph_objects as go

from context_manager import ConversationContext
import vision_analyzer as va

st.set_page_config(page_title="Multimodal AI Assistant (Gemini)", page_icon="🤖", layout="wide")

# ---------------------------------------------------------------- session
if "context" not in st.session_state:
    st.session_state.context = ConversationContext()
if "messages" not in st.session_state:
    st.session_state.messages = []  # display-only list of dicts

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        help="Get a free key at https://aistudio.google.com/apikey (no credit card needed).",
    )
    model_name = st.selectbox(
        "Model",
        ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
        index=0,
        help="Flash / Flash-Lite are covered by the free tier. Pro has a much lower free daily quota.",
    )

    st.divider()
    st.subheader("📊 Quality Dashboard")
    ctx = st.session_state.context
    if ctx.quality_history:
        st.metric("Average Response Quality", f"{ctx.average_quality():.0f}%")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=ctx.quality_history,
                mode="lines+markers",
                line=dict(color="#6c5ce7"),
                marker=dict(size=8),
            )
        )
        fig.update_layout(
            height=220,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, 100], title="Score"),
            xaxis=dict(title="Turn"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No responses yet - quality scores will appear here.")

    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.context = ConversationContext()
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------------------------------- header
st.title("🤖 Multimodal AI Assistant")
st.caption("Text + Vision reasoning powered by Google Gemini (free tier) - no Anthropic key required.")

if not api_key:
    st.info(
        "Enter your free Gemini API key in the sidebar to get started. "
        "Grab one at https://aistudio.google.com/apikey"
    )

# ---------------------------------------------------------- render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image") is not None:
            st.image(msg["image"], width=280)
        st.markdown(msg["text"])
        if msg["role"] == "assistant" and msg.get("quality") is not None:
            score = msg["quality"]["score"]
            badge = "🟢" if score >= 80 else ("🟡" if score >= 50 else "🔴")
            with st.expander(f"{badge} Response quality: {score}%  ·  intent: {msg.get('intent', 'general')}"):
                for check, passed in msg["quality"]["checks"].items():
                    st.write(("✅ " if passed else "❌ ") + check.replace("_", " ").title())
                st.caption(f"Word count: {msg['quality']['word_count']}")

# --------------------------------------------------------------- new input
uploaded_image = st.file_uploader("📷 Attach an image (optional)", type=["png", "jpg", "jpeg", "webp"])
user_text = st.chat_input("Ask something about the image, or just chat...")

if user_text:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar first.")
        st.stop()

    pil_image = Image.open(uploaded_image) if uploaded_image is not None else None

    ctx = st.session_state.context
    intent = ctx.detect_intent(user_text)

    # --- show + record the user's turn ---
    st.session_state.messages.append(
        {"role": "user", "text": user_text, "image": pil_image, "intent": intent}
    )
    ctx.add_user_turn(user_text, has_image=pil_image is not None, intent=intent)

    with st.chat_message("user"):
        if pil_image is not None:
            st.image(pil_image, width=280)
        st.markdown(user_text)

    # --- build prompt + call Gemini ---
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            history_ctx = ctx.history_summary()
            prompt = va.build_prompt(user_text, intent, history_ctx)
            processed_image = va.encode_image(pil_image) if pil_image is not None else None

            try:
                response_text = va.call_gemini(api_key, model_name, prompt, processed_image)
            except Exception as e:
                response_text = f"⚠️ Error calling Gemini API: {e}"

            quality = va.validate_response(response_text, intent)

        st.markdown(response_text)
        score = quality["score"]
        badge = "🟢" if score >= 80 else ("🟡" if score >= 50 else "🔴")
        with st.expander(f"{badge} Response quality: {score}%  ·  intent: {intent}"):
            for check, passed in quality["checks"].items():
                st.write(("✅ " if passed else "❌ ") + check.replace("_", " ").title())
            st.caption(f"Word count: {quality['word_count']}")

    # --- record assistant turn (+ lightweight image memory) ---
    image_desc = f"[Image discussed in turn about: {user_text[:60]}]" if pil_image is not None else None
    ctx.add_assistant_turn(response_text, score, image_description=image_desc)
    st.session_state.messages.append(
        {"role": "assistant", "text": response_text, "quality": quality, "intent": intent}
    )
