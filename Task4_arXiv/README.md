# 📚 arXiv Computer Science Expert Chatbot

A domain-expert chatbot over the [Cornell arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv),
scoped to computer science (`cs.*`) categories. It searches papers,
generates summaries, explains concepts, and answers follow-up questions -
grounded in retrieved paper excerpts (Retrieval-Augmented Generation) using
an **open-source Hugging Face model**, no paid API required.

---
## 🗂️ Project Structure
```
arxiv_expert_chatbot/
├── app.py                     # Streamlit UI (3 tabs: search, chat, explore)
├── data_loader.py              # loads real dataset or bundled demo sample
├── text_processing.py          # TF-IDF keyword extraction + TextRank summarization
├── retrieval.py                 # semantic (or TF-IDF fallback) search index
├── llm_engine.py                 # Hugging Face Transformers: summarize + explain
├── conversation.py               # multi-turn context + follow-up query expansion
├── visualize.py                   # category/trend charts + 2D concept map
├── data/
│   └── sample_arxiv_cs.json        # bundled 30-paper demo dataset (works offline)
├── scripts/
│   └── prepare_dataset.py           # filters the full Kaggle dump to a CS subset
├── requirements.txt
└── README.md
```

---
## 🧠 Architecture
```
User query (search or chat)
         ↓
Follow-up detection & query expansion (conversation.py)
         ↓
Retrieval: semantic search over paper title+abstract (retrieval.py)
   → sentence-transformers embeddings, or TF-IDF fallback if offline
         ↓
Retrieved paper excerpts = grounding context
         ↓
LLM generation (llm_engine.py, Hugging Face Transformers / Flan-T5)
   - summarize()        → per-paper abstractive summary
   - explain()           → RAG-grounded answer to a question
   - explain_concept()    → standalone concept explanation
         ↓
Answer + cited sources shown in Streamlit chat
```

Classic NLP (TF-IDF keyword extraction, TextRank extractive summarization
in `text_processing.py`) runs alongside the LLM rather than being replaced
by it - fast, deterministic, and useful even before an LLM call is made.

---
## 🚀 Setup & Run

```bash
# 1. Create the folder and copy all project files into it
cd arxiv_expert_chatbot

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run (works immediately using the bundled 30-paper demo dataset)
streamlit run app.py
```

On first use of the chat/search features, `sentence-transformers` and
`transformers` will each download a small model from the Hugging Face Hub
(~80MB embedding model, ~250MB Flan-T5-Base) - this needs internet access
once; afterwards models are cached locally.

---
## 📥 Using the real Kaggle arXiv dataset

The app runs out of the box on a bundled 30-paper demo sample so you can
try it immediately. To search the **real** dataset (2.7M+ papers, filtered
to CS):

```bash
# 1. Get a free Kaggle API token: https://www.kaggle.com/settings -> Create New Token
#    Save the downloaded kaggle.json to ~/.kaggle/kaggle.json

pip install kaggle

# 2. Download the dataset (~4GB compressed)
kaggle datasets download -d Cornell-University/arxiv -p ./raw_data --unzip

# 3. Filter it down to a CS subset (streams the file, doesn't load it all into RAM)
python scripts/prepare_dataset.py --input raw_data/arxiv-metadata-oai-snapshot.json
```

This writes `data/arxiv_cs_subset.parquet`, which `app.py` picks up
automatically on the next run (the sidebar will confirm it's no longer
using the demo sample). By default it keeps up to 2,000 CS papers per
year to keep the corpus fast to search on a laptop - adjust
`--max-per-year` (or set it to `0` for no cap) in `prepare_dataset.py`.

---
## ⚙️ Model choice

| Model | Params | Notes |
|---|---|---|
| `google/flan-t5-base` (default) | ~250M | Fast on CPU, good default |
| `google/flan-t5-large` | ~780M | Better quality, noticeably slower on CPU |

Switch between them from the sidebar. For GPU machines, larger
instruction-tuned models (e.g. `microsoft/Phi-3-mini-4k-instruct`) can be
swapped into `llm_engine.py`'s `DEFAULT_MODEL` / `MODEL_OPTIONS` for
better answer quality.

---
## ✅ Feature coverage

| Requirement | Implementation |
|---|---|
| Domain-specific expert chatbot | Scoped to arXiv `cs.*` categories |
| Trained/grounded on arXiv subset | `data_loader.py` + `prepare_dataset.py` |
| Information extraction | TF-IDF keyword/key-phrase extraction |
| Summarization | TextRank extractive + LLM abstractive summaries |
| Open-source LLM for explanation | Hugging Face Transformers (Flan-T5) |
| Follow-up question handling | `conversation.py` query expansion + chat history in prompt |
| Paper searching | Semantic search tab with relevance ranking |
| Concept visualization | Category distribution, publication trend, 2D concept map |

---
## ⚠️ Known limitations
- The Kaggle dataset contains **metadata only** (titles/abstracts), not
  full paper text - summaries and answers are grounded in abstracts, not
  entire papers.
- Flan-T5-Base is a small model; answers are reasonable but not as fluent
  as a large chat model. Swap in `flan-t5-large` or a bigger model from
  the Hub if your hardware allows.
- The TF-IDF fallback retrieval (used automatically if the embedding model
  can't be downloaded) is keyword-based rather than truly semantic, so
  search quality is a bit lower without internet access on first run.
