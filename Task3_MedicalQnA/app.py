"""
app.py — Medical Q&A Chatbot using MedQuAD Dataset
TF-IDF retrieval + Medical NER + Streamlit UI
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

from data_loader import create_sample_csv, parse_medquad_xml
from retriever import MedicalRetriever
from entity_recognizer import (
    recognize, ENTITY_COLORS, ENTITY_ICONS
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Q&A Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.user-bubble {
    background: #3b82f6; color: white;
    padding: 10px 14px; border-radius: 18px 18px 4px 18px;
    margin: 6px 0; max-width: 75%; float: right; clear: both; font-size:14px;
}
.bot-bubble {
    background: #f0fdf4; color: #1a1a1a; border: 1px solid #bbf7d0;
    padding: 12px 16px; border-radius: 18px 18px 18px 4px;
    margin: 6px 0; max-width: 85%; float: left; clear: both;
    font-size:14px; line-height:1.6;
}
.entity-tag {
    display:inline-block; padding:2px 8px; border-radius:12px;
    font-size:12px; font-weight:600; margin:2px;
}
.score-bar {
    background:#e5e7eb; border-radius:4px; height:6px; margin:4px 0;
}
.score-fill { border-radius:4px; height:6px; }
.disclaimer {
    background:#fef3c7; border:1px solid #f59e0b;
    border-radius:8px; padding:10px; font-size:12px; color:#92400e; margin:10px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_log" not in st.session_state:
    st.session_state.query_log = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

# ── Auto-build index on startup ────────────────────────────────────────────────
if not st.session_state.index_ready:
    r = MedicalRetriever()
    if r.load_index():
        st.session_state.retriever = r
        st.session_state.index_ready = True
    else:
        # Auto-create sample CSV and build index silently
        if not os.path.exists("medquad_data.csv"):
            create_sample_csv()
        r.build_index()
        st.session_state.retriever = r
        st.session_state.index_ready = True

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 Medical Q&A")
    st.caption("MedQuAD · TF-IDF Retrieval · Medical NER")
    st.divider()

    # Dataset info
    st.subheader("📂 Dataset")
    csv_exists = os.path.exists("medquad_data.csv")
    if csv_exists:
        df_info = pd.read_csv("medquad_data.csv")
        st.success(f"✅ {len(df_info):,} QA pairs loaded")
        st.caption("Source: MedQuAD / NIH")
    else:
        st.info("Using built-in sample dataset")

    # Load full MedQuAD option
    with st.expander("🔄 Load Full MedQuAD Dataset"):
        repo_path = st.text_input(
            "Path to cloned MedQuAD repo:",
            placeholder="/home/user/MedQuAD",
        )
        st.caption("Clone: `git clone https://github.com/abachaa/MedQuAD`")
        if st.button("Parse & Rebuild Index", use_container_width=True):
            if repo_path and os.path.exists(repo_path):
                with st.spinner("Parsing XML files..."):
                    count = parse_medquad_xml(repo_path)
                with st.spinner("Building TF-IDF index..."):
                    r2 = MedicalRetriever()
                    r2.build_index()
                    st.session_state.retriever = r2
                st.success(f"✅ {count:,} QA pairs indexed!")
                st.rerun()
            else:
                st.error("Invalid path. Clone MedQuAD first.")

    st.divider()

    # Search settings
    st.subheader("⚙️ Settings")
    top_k = st.slider("Results to retrieve", 1, 8, 3)
    min_score = st.slider("Min relevance score", 0.0, 1.0, 0.05, 0.01)

    st.divider()

    # Analytics
    if st.session_state.query_log:
        st.subheader("📊 Analytics")
        log = st.session_state.query_log
        st.metric("Total Queries", len(log))

        # Entity distribution
        all_entities = []
        for entry in log:
            for label, terms in entry.get("entities", {}).items():
                all_entities.extend([label] * len(terms))

        if all_entities:
            ec = pd.Series(all_entities).value_counts()
            fig = go.Figure(go.Pie(
                labels=[f"{ENTITY_ICONS.get(l,'•')} {l}" for l in ec.index],
                values=ec.values, hole=0.5,
            ))
            fig.update_layout(
                height=200, margin=dict(t=10, b=10, l=0, r=0),
                showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Avg score
        scores = [e.get("top_score", 0) for e in log]
        st.metric("Avg Relevance", f"{sum(scores)/len(scores):.2f}")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_log = []
        st.rerun()

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("🏥 Medical Q&A Chatbot")
st.caption("Powered by MedQuAD dataset · TF-IDF retrieval · Medical entity recognition")

st.markdown("""
<div class="disclaimer">
⚠️ <strong>Disclaimer:</strong> This chatbot provides general health information
from NIH MedQuAD sources only. It is NOT a substitute for professional medical
advice, diagnosis, or treatment. Always consult a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)

# ── Chat ───────────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">{msg["content"]}</div>'
            f'<div style="clear:both"></div>',
            unsafe_allow_html=True,
        )

    elif msg["role"] == "bot":
        st.markdown(
            f'<div class="bot-bubble">{msg["content"]}</div>'
            f'<div style="clear:both"></div>',
            unsafe_allow_html=True,
        )

        # Show sources
        if msg.get("results"):
            with st.expander(f"📚 {len(msg['results'])} Source(s) from MedQuAD"):
                for i, res in enumerate(msg["results"], 1):
                    pct = int(res["score"] * 100)
                    color = "#10b981" if pct > 40 else "#f59e0b" if pct > 20 else "#6b7280"
                    st.markdown(f"""
<div style="background:#f8fafc;border-radius:8px;padding:10px;margin:6px 0;border:1px solid #e2e8f0">
<strong>#{i}</strong> — <em>{res['focus'] or 'Medical'}</em>
<span style="float:right;color:{color};font-size:12px;font-weight:600">{pct}% match</span><br>
<small style="color:#6b7280">{res['question']}</small>
<div class="score-bar"><div class="score-fill" style="width:{pct}%;background:{color}"></div></div>
</div>""", unsafe_allow_html=True)

        # Show entities
        if msg.get("entities"):
            st.markdown("**🔍 Entities Detected:**")
            cols = st.columns(len(msg["entities"]))
            for i, (label, terms) in enumerate(msg["entities"].items()):
                with cols[i]:
                    icon = ENTITY_ICONS.get(label, "•")
                    color = ENTITY_COLORS.get(label, "#f3f4f6")
                    tags = " ".join([
                        f'<span class="entity-tag" style="background:{color}">{t}</span>'
                        for t in terms[:3]
                    ])
                    st.markdown(
                        f'<div style="font-size:12px"><strong>{icon} {label}</strong><br>{tags}</div>',
                        unsafe_allow_html=True,
                    )

# ── Input ──────────────────────────────────────────────────────────────────────
st.divider()
with st.form("qa_form", clear_on_submit=True):
    col_in, col_btn = st.columns([5, 1])
    user_q = col_in.text_input(
        "Ask",
        placeholder="e.g. What are the symptoms of diabetes?",
        label_visibility="collapsed",
    )
    submitted = col_btn.form_submit_button("Ask 🔍", type="primary", use_container_width=True)


def handle_query(query: str):
    retriever: MedicalRetriever = st.session_state.retriever

    # NER on query
    ner = recognize(query)
    entity_summary = ner.summary()

    # Retrieve
    results = retriever.search(query, top_k=top_k)
    filtered = [r for r in results if r.score >= min_score]

    if not filtered:
        bot_reply = (
            "I couldn't find a relevant answer in the MedQuAD dataset for your question. "
            "Please try rephrasing, or consult a medical professional for accurate advice."
        )
        result_dicts = []
    else:
        best = filtered[0]
        prefix = f"**Regarding {best.focus}:**\n\n" if best.focus else ""
        bot_reply = prefix + best.answer
        result_dicts = [
            {"question": r.question, "answer": r.answer,
             "focus": r.focus, "score": r.score}
            for r in filtered
        ]

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({
        "role": "bot", "content": bot_reply,
        "results": result_dicts, "entities": entity_summary,
    })
    st.session_state.query_log.append({
        "query": query, "entities": entity_summary,
        "top_score": filtered[0].score if filtered else 0,
        "timestamp": datetime.now().isoformat(),
    })


if submitted and user_q.strip():
    handle_query(user_q.strip())
    st.rerun()

# ── Example questions ──────────────────────────────────────────────────────────
st.markdown("**💡 Try these:**")
examples = [
    "What are the symptoms of diabetes?",
    "How is hypertension treated?",
    "What causes asthma?",
    "What are early signs of Alzheimer's?",
    "How is pneumonia treated?",
    "What is heart disease?",
    "What are COPD symptoms?",
    "How is depression treated?",
    "What causes kidney disease?",
    "What are stroke symptoms?",
    "How is anemia treated?",
    "What is COVID-19?",
]
cols = st.columns(4)
for i, ex in enumerate(examples):
    if cols[i % 4].button(ex, key=f"ex_{i}", use_container_width=True):
        handle_query(ex)
        st.rerun()
