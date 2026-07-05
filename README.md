# 🧠 Elevanceskills — AI/ML Internship Projects

> **Akash Sikarwar** | B.Tech Information Technology, MMMUT Gorakhpur (2024–2028)
> GitHub: [@Achiever199](https://github.com/Achiever199)

A series of six AI/ML projects covering the spectrum of modern NLP and applied AI: retrieval-augmented generation, multimodal reasoning, domain-specific medical Q&A, scientific-literature question answering, sentiment-aware dialogue, and multilingual conversation — each built and deployed independently as a Streamlit app.

---

## 📋 Project Overview

| # | Project | One-line summary | Tech Stack | Live Demo |
|---|---|---|---|---|
| [Task 1](#task-1--dynamic-knowledge-base-system) | Dynamic Knowledge Base System | Self-updating RAG pipeline that re-indexes only new/changed content from files, web pages, and RSS feeds on a schedule | ChromaDB, SQLite, APScheduler, Python | Local/Docker deployment |
| [Task 2](#task-2--multimodal-ai-assistant-text--vision) | Multimodal AI Assistant | Text + image chatbot with intent detection, ambiguity handling, and 5-check response validation | Streamlit, Google Gemini Vision API | [task2appdo.streamlit.app](https://task2appdo.streamlit.app/) |
| [Task 3](#task-3--medical-qa-chatbot-medquad-dataset) | Medical Q&A Chatbot | Domain-specific medical Q&A over the MedQuAD dataset with retrieval + rule-based medical NER | Streamlit, scikit-learn (TF-IDF) | [task3med.streamlit.app](https://task3med.streamlit.app/) |
| [Task 4](#task-4--arxiv-computer-science-expert-chatbot) | arXiv CS Expert Chatbot | RAG chatbot over arXiv computer-science papers with search, summarization, and follow-up handling | Streamlit, Hugging Face Transformers (Flan-T5), sentence-transformers | [task4arxiv.streamlit.app](https://task4arxiv.streamlit.app/) |
| [Task 5](#task-5--sentiment-aware-customer-support-chatbot) | Sentiment-Aware Chatbot | Customer support bot that detects sentiment/emotion and adapts tone, with live analytics dashboard | Streamlit, VADER + TextBlob ensemble | [sentimentchatbot...streamlit.app](https://sentimentchatbot-kcft6jnttnrdhm5wqy9qqy.streamlit.app/) |
| [Task 6](#task-6--multilingual-conversational-chatbot) | Multilingual Chatbot | Conversational bot supporting 10 languages with automatic detection, translation, and context retention across language switches | Streamlit, langdetect, deep-translator | [task6multilingualchatbot...streamlit.app](https://task6multilingualchatbot-vjadnwqty5txfpysrmavni.streamlit.app/) |

Each project lives in its own folder with a complete standalone README — this document summarizes all six and links out to the details.

---

## Task 1 — Dynamic Knowledge Base System
📁 [`Task_1/`](./Task_1)

A production-ready system that keeps a chatbot's vector database continuously up to date with new information pulled from external sources (files, web pages, RSS feeds) — without re-embedding content that hasn't changed.

**Pipeline:** `Sources (file/web/rss) → pluggable Loaders → chunk + content-hash → SQLite manifest (change detection) → Chroma vector store (upsert only new/changed) → chatbot (retrieval, optional LLM-generated answer)`

**Key design points:**
- **Change detection, not blind re-indexing** — every document is content-hashed; unchanged documents are skipped entirely on each run.
- **Stale-content cleanup** — when a document changes, its old chunks are deleted before new ones are inserted; removed documents have their vectors purged.
- **Per-source scheduling** — each source in `config.yaml` has its own APScheduler `interval` or `cron` schedule (e.g. an RSS feed refreshed hourly, a static folder rescanned nightly).
- **Pluggable sources** — `file`, `web`, and `rss` loaders included; new source types are a ~30-line class.
- **No required external API** — embeddings default to Chroma's bundled ONNX MiniLM model (no torch/GPU needed); answer generation upgrades automatically to LLM-composed answers if `ANTHROPIC_API_KEY` is set.

**Run it:**
```bash
pip install -r requirements.txt
python main.py update-now   # one-off index build
python main.py chat         # ask questions
python main.py serve        # continuous scheduler — the "dynamic expansion" part
```
Also ships with a `Dockerfile` + `docker-compose.yml` for containerized deployment with bind-mounted persistent storage.

---

## Task 2 — Multimodal AI Assistant (Text + Vision)
📁 [`Task2_Multimodal_Assistant/`](./Task2_Multimodal_Assistant) · 🔗 [Live demo](https://task2appdo.streamlit.app/)

A text + image chatbot that analyzes visual content, extracts relevant information, maintains conversational context, and generates evidence-based responses with ambiguity handling and response validation — built on **Google Gemini's free API tier** (no Anthropic key required).

**Architecture:** image + text input → intent detection (9 intents: describe, identify, extract, compare, analyze, count, locate, sentiment, general) → prompt built from conversation history → Gemini Vision (`gemini-2.5-flash`) → 5-check response validator (observations, reasoning, uncertainty handling, substance, no hallucination) → quality-scored answer with a trend dashboard.

**Key features:**
- Multi-turn context retention via a rolling conversation history summary
- Ambiguity handling enforced through system-prompt confidence flags ("appears to", "I'm not certain")
- Per-response quality badge (0–100%) plus a sidebar trend chart
- Multi-image support — prior images referenced by short description in later turns

**Run it:**
```bash
pip install -r requirements.txt
streamlit run app.py
```
Free Gemini key: https://aistudio.google.com/apikey — pasted into the sidebar, kept in-session only.

---

## Task 3 — Medical Q&A Chatbot (MedQuAD Dataset)
📁 [`Task3_MedicalQnA/`](./Task3_MedicalQnA) · 🔗 [Live demo](https://task3med.streamlit.app/)

A specialized medical question-answering chatbot built on the [MedQuAD dataset](https://github.com/abachaa/MedQuAD), combining TF-IDF retrieval with rule-based medical entity recognition.

**Pipeline:** `User query → TF-IDF vectorization → cosine similarity → top-k results → best answer`, alongside a dictionary-based NER pass over 150+ medical terms across 5 entity types (🤒 symptom, 🦠 disease, 💊 treatment, 💉 medication, 🫀 body part).

**Why TF-IDF instead of embeddings here:** zero network dependency, sub-second indexing, and medical QA pairs use specific-enough terminology that TF-IDF performs well without a 400MB model download.

**Results:** ships with a curated 50-pair sample (15 diseases) that works offline out of the box; optionally loads the full MedQuAD corpus (16,407 QA pairs from 12 NIH sources) via a sidebar path input. Index builds in <1 second with sub-100ms query response.

**Run it:**
```bash
pip install streamlit scikit-learn pandas numpy plotly lxml
streamlit run app.py
```
> ⚠️ Provides general health information from NIH sources only — not a substitute for professional medical advice.

---

## Task 4 — arXiv Computer Science Expert Chatbot
📁 [`Task4_arXiv/`](./Task4_arXiv) · 🔗 [Live demo](https://task4arxiv.streamlit.app/)

A domain-expert chatbot over the [Cornell arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv), scoped to computer-science (`cs.*`) categories. Searches papers, generates summaries, explains concepts, and handles follow-up questions — grounded in retrieved paper excerpts via Retrieval-Augmented Generation, using an **open-source Hugging Face model** (no paid API).

**Pipeline:** query → follow-up detection & query expansion → semantic search (sentence-transformers embeddings, with automatic TF-IDF fallback if offline) → retrieved excerpts as grounding context → Hugging Face Transformers (`flan-t5-base`/`large`) for abstractive summarization and RAG-grounded explanation → answer + cited sources.

**Key features:**
- Classic NLP (TF-IDF keyword extraction, TextRank extractive summarization) runs alongside the LLM rather than being replaced by it
- Three tabs: 🔍 Search Papers, 💬 Ask the Expert (chat), 📊 Explore the Field (category distribution, publication trend, 2D concept map)
- `scripts/prepare_dataset.py` streams and filters the full ~4GB Kaggle dump down to a CS-only subset without loading it all into memory
- Ships with a bundled 30-paper demo dataset so it works before the full Kaggle dataset is downloaded

**Run it:**
```bash
pip install -r requirements.txt
streamlit run app.py
```
See the task README for the full Kaggle-dataset download/prep steps and a CPU-only torch install note (avoids a common multi-GB CUDA download failure).

---

## Task 5 — Sentiment-Aware Customer Support Chatbot
📁 [`Task5_Sentiment_Chatbot/`](./Task5_Sentiment_Chatbot) · 🔗 [Live demo](https://sentimentchatbot-kcft6jnttnrdhm5wqy9qqy.streamlit.app/)

Integrates real-time sentiment analysis into a customer-support chatbot so responses match how the user is actually feeling, rather than replying with the same generic tone regardless of context.

**Model:** a weighted ensemble of **VADER** (70%) + **TextBlob** (30%) —
`ensemble_score = 0.70 × VADER_compound + 0.30 × TextBlob_polarity`, thresholded into positive/negative/neutral, plus keyword-based detection of 7 fine-grained emotions (frustrated, angry, sad, anxious, happy, grateful, confused).

**Results:** 83.3% accuracy on a 12-case hand-labeled test set — outperforming VADER alone (~75%) and TextBlob alone (~67%). Known edge cases: "disappointed" scores mildly negative in VADER and "confused" is misread as negative sentiment rather than neutral confusion.

**Features:** live chat with a sentiment badge per message, real-time analytics sidebar (donut chart, score timeline, emotion breakdown), 1–5 star satisfaction rating, automatic escalation offer for high-confidence negative messages.

**Run it:**
```bash
pip install --timeout 300 streamlit vaderSentiment textblob pandas plotly nltk
python -m textblob.download_corpora
streamlit run app.py
```

---

## Task 6 — Multilingual Conversational Chatbot
📁 [`Task6_Multilingual_Chatbot/`](./Task6_Multilingual_Chatbot) · 🔗 [Live demo](https://task6multilingualchatbot-vjadnwqty5txfpysrmavni.streamlit.app/)

Supports multilingual conversations across **10 languages** (English, Hindi, French, Spanish, German, Chinese, Arabic, Portuguese, Japanese, Russian — including right-to-left Arabic) while preserving context, intent, and conversational continuity across language switches.

**Pipeline:** `input (any language) → langdetect → translate to English (deep-translator) → intent detection → generate English response → translate back → context update (tracks language switches + topics) → display with language badge`

**Key features:**
- Language-switch detection with an acknowledgement banner in the new language
- Mixed-language input handling (splits and detects per segment)
- Context retention across turns and language switches via a `ConversationContext` dataclass
- Live analytics sidebar — language distribution pie chart + timeline

**Run it:**
```bash
pip install streamlit langdetect deep-translator pandas plotly
streamlit run app.py
```

---

## 🛠️ Tech Stack Summary

| Category | Technologies used across the six projects |
|---|---|
| UI | Streamlit (all six projects) |
| LLM / Generation | Google Gemini Vision (Task 2), Hugging Face Transformers / Flan-T5 (Task 4), Anthropic API — optional (Task 1) |
| Retrieval / Vector Search | ChromaDB (Task 1), sentence-transformers + TF-IDF fallback (Task 4), TF-IDF (Task 3) |
| Classic NLP | TextRank summarization + TF-IDF keyword extraction (Task 4), rule-based medical NER (Task 3), VADER + TextBlob (Task 5), langdetect (Task 6) |
| Scheduling / Infra | APScheduler, SQLite, Docker (Task 1) |
| Visualization | Plotly across Tasks 2, 3, 5, 6; Plotly + PCA concept maps in Task 4 |
| Language | Python 3.12 |

---

## 👤 Author

**Akash Sikarwar**
🎓 B.Tech Information Technology — MMMUT Gorakhpur (2024–2028)
🐙 GitHub: [@Achiever199](https://github.com/Achiever199)
