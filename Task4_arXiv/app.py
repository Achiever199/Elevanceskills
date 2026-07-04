"""
app.py
------
Streamlit UI for the arXiv CS Domain-Expert Chatbot.

Three tabs:
  1. Search Papers      - semantic/TF-IDF search + per-paper summaries
  2. Ask the Expert      - RAG chat grounded in retrieved paper excerpts,
                           with follow-up question handling
  3. Explore the Field   - category distribution, publication trends,
                           and a 2D concept map of the corpus
"""

import streamlit as st

import data_loader as dl
import text_processing as tp
import visualize as viz
from conversation import ConversationContext
from llm_engine import LLMEngine
from retrieval import PaperIndex

st.set_page_config(page_title="arXiv CS Expert Chatbot", page_icon="📚", layout="wide")

MODEL_OPTIONS = {
    "Flan-T5 Base (fast, ~250M params - recommended for CPU)": "google/flan-t5-base",
    "Flan-T5 Large (better quality, ~780M params, slower on CPU)": "google/flan-t5-large",
}


# ---------------------------------------------------------------- caching
@st.cache_data(show_spinner="Loading paper corpus...")
def get_data():
    df = dl.load_papers()
    return df


@st.cache_resource(show_spinner="Building search index (first run downloads a small embedding model)...")
def get_index(_df_hash: int, df):
    return PaperIndex(df)


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str):
    return LLMEngine(model_name)


# ---------------------------------------------------------------- session
if "conversation" not in st.session_state:
    st.session_state.conversation = ConversationContext()
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []  # [{"role", "text", "sources": [...] }]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("⚙️ Settings")

    model_label = st.selectbox("LLM model (Hugging Face Transformers)", list(MODEL_OPTIONS.keys()))
    model_name = MODEL_OPTIONS[model_label]

    df_full = get_data()
    all_categories = dl.get_available_categories(df_full)
    selected_categories = st.multiselect(
        "Filter by CS subfield",
        all_categories,
        default=[],
        help="Leave empty to search across all loaded subfields.",
    )

    top_k = st.slider("Papers to retrieve per query", min_value=2, max_value=8, value=4)

    st.divider()
    if dl.is_using_sample_data():
        st.warning(
            "Using the bundled **demo sample** (30 papers). Run "
            "`scripts/prepare_dataset.py` after downloading the full Kaggle "
            "arXiv dataset to search the real corpus."
        )
    else:
        st.success(f"Loaded prepared arXiv CS subset: {len(df_full):,} papers.")

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.conversation.reset()
        st.session_state.chat_display = []
        st.rerun()

df = dl.filter_by_categories(df_full, selected_categories)
if df.empty:
    st.error("No papers match the selected subfield filters. Try clearing some filters.")
    st.stop()

index = get_index(hash(tuple(df["id"])), df)
llm = get_llm(model_name)

st.title("📚 arXiv Computer Science Expert Chatbot")
st.caption(
    f"Grounded in {len(df):,} paper record(s) "
    f"({'demo sample' if dl.is_using_sample_data() else 'prepared arXiv subset'}) · "
    f"retrieval backend: **{index.backend}** · LLM: **{model_name}**"
)

tab_search, tab_chat, tab_explore = st.tabs(["🔍 Search Papers", "💬 Ask the Expert", "📊 Explore the Field"])

# =========================================================== SEARCH TAB
with tab_search:
    st.subheader("Search the paper corpus")
    query = st.text_input("Search by topic, method, or keywords", key="search_query")

    if query:
        results = index.search_papers(query, top_k=top_k)
        if results.empty:
            st.info("No matching papers found. Try a broader query or clear subfield filters.")
        for _, paper in results.iterrows():
            with st.expander(f"📄 {paper['title']}  ·  relevance {paper['relevance']:.2f}"):
                st.markdown(
                    f"**Authors:** {paper.get('authors', 'N/A')}  \n"
                    f"**Categories:** `{paper.get('categories', 'N/A')}`  ·  "
                    f"**Year:** {paper.get('year', 'N/A')}"
                )
                st.markdown("**Abstract:**")
                st.write(paper["abstract"])

                quick_summary = tp.extractive_summary(paper["abstract"], num_sentences=2)
                st.markdown("**Quick extractive summary:**")
                st.info(quick_summary)

                keywords = tp.extract_keywords(paper["abstract"], top_n=6)
                if keywords:
                    st.markdown("**Key terms:** " + ", ".join(f"`{k}`" for k in keywords))

                if st.button("🧠 Generate LLM summary", key=f"llm_sum_{paper['id']}"):
                    with st.spinner("Generating summary with the LLM..."):
                        llm_summary = llm.summarize(paper["abstract"])
                    st.markdown("**LLM abstractive summary:**")
                    st.success(llm_summary)
    else:
        st.caption("Enter a query above, e.g. *\"contrastive learning for images\"* or *\"federated optimization\"*.")

# =========================================================== CHAT TAB
with tab_chat:
    st.subheader("Ask the expert")
    st.caption(
        "Answers are grounded in retrieved paper excerpts (RAG). Follow-up "
        "questions like *\"why does that help?\"* reuse the topic from your "
        "previous message automatically."
    )

    for msg in st.session_state.chat_display:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            if msg.get("sources"):
                with st.expander("📎 Sources used"):
                    for s in msg["sources"]:
                        st.markdown(f"- {s}")

    user_question = st.chat_input("Ask about a concept, method, or paper...")

    if user_question:
        conv = st.session_state.conversation
        st.session_state.chat_display.append({"role": "user", "text": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        expanded_query = conv.expand_query(user_question)
        conv.add_user_turn(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant papers and generating an answer..."):
                hits = index.search_papers(expanded_query, top_k=top_k)

                if hits.empty:
                    answer = (
                        "I couldn't find any papers in the current corpus relevant to that "
                        "question. Try rephrasing, or widen the subfield filters in the sidebar."
                    )
                    source_lines = []
                else:
                    context_blocks = []
                    source_lines = []
                    for _, paper in hits.iterrows():
                        context_blocks.append(f"[{paper['title']}]: {paper['abstract']}")
                        source_lines.append(f"**{paper['title']}** ({paper.get('year', 'N/A')})")
                    context = "\n\n".join(context_blocks)
                    history = conv.history_text(max_turns=4)
                    answer = llm.explain(user_question, context, history=history)

            st.markdown(answer)
            if source_lines:
                with st.expander("📎 Sources used"):
                    for s in source_lines:
                        st.markdown(f"- {s}")

        conv.add_assistant_turn(answer, sources=source_lines)
        st.session_state.chat_display.append({"role": "assistant", "text": answer, "sources": source_lines})

# =========================================================== EXPLORE TAB
with tab_explore:
    st.subheader("Explore the field")

    col1, col2 = st.columns(2)
    with col1:
        fig_cat = viz.category_distribution_chart(df)
        if fig_cat:
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.caption("Not enough category data to plot.")
    with col2:
        fig_year = viz.papers_per_year_chart(df)
        if fig_year:
            st.plotly_chart(fig_year, use_container_width=True)
        else:
            st.caption("Not enough year data to plot.")

    st.markdown("---")
    st.markdown("**Concept map** - papers positioned by textual similarity, colored by subfield.")
    fig_map = viz.concept_map(df)
    if fig_map:
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.caption("Need at least 3 papers to draw a concept map.")
