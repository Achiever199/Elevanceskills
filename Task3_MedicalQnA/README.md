# 🏥 Medical Q&A Chatbot — MedQuAD Dataset

> **Internship Project**
> **Live Preview:https://task3med.streamlit.app/**
> Task 3 | Built by [Akash Sikarwar](https://github.com/Achiever199)

---

## 📌 Problem Statement

Build a specialized medical Q&A chatbot using the MedQuAD dataset with a retrieval mechanism, medical entity recognition, and a Streamlit UI.

---

## 🗂️ Project Structure

```
Task3_Medical_QA/
├── app.py                  # Streamlit UI
├── data_loader.py          # MedQuAD XML parser + built-in 50-pair sample
├── retriever.py            # TF-IDF retrieval engine (sklearn)
├── entity_recognizer.py    # Medical NER (symptoms, diseases, treatments)
├── requirements.txt
└── README.md
```

---

## 🧠 Methodology

### Retrieval Pipeline
```
User Query → TF-IDF Vectorization → Cosine Similarity → Top-K Results → Best Answer
```

| Component | Technology | Why |
|---|---|---|
| Vectorizer | TF-IDF (sklearn) | Lightweight, no downloads, instant |
| Similarity | Cosine similarity | Standard IR metric |
| NER | Rule-based dictionary | 150+ medical terms, 5 entity types |
| UI | Streamlit | Simple, interactive |

### Why TF-IDF over sentence-transformers?
- **Zero network dependency** — no 400MB model downloads
- **Instant indexing** — builds in <1 second vs minutes
- **Sufficient accuracy** — medical QA pairs have specific terminology that TF-IDF handles well
- **Production-ready** — used in real medical IR systems

### Medical NER — 5 Entity Types

| Type | Icon | Examples |
|---|---|---|
| Symptom | 🤒 | fever, chest pain, fatigue |
| Disease | 🦠 | diabetes, hypertension, cancer |
| Treatment | 💊 | chemotherapy, dialysis, surgery |
| Medication | 💉 | metformin, aspirin, insulin |
| Body Part | 🫀 | heart, lung, kidney |

---

## 🚀 Setup & Run

```bash
# 1. Create folder and go in
mkdir Task3_Medical_QA && cd Task3_Medical_QA

# 2. Virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install (all lightweight!)
pip install streamlit scikit-learn pandas numpy plotly lxml

# 4. Run — auto-loads built-in dataset on first launch
streamlit run app.py
```

**Optional: Load full MedQuAD (16,407 QA pairs)**
```bash
git clone https://github.com/abachaa/MedQuAD ~/Desktop/MedQuAD
# Then enter the path in the sidebar "Load Full MedQuAD Dataset" section
```

---

## 📊 Results

- Built-in sample: 50 curated NIH QA pairs across 15 diseases
- Full MedQuAD: 16,407 QA pairs from 12 NIH websites
- Index builds in <1 second (TF-IDF)
- Sub-100ms query response time
- 5-type medical entity recognition

---

## ⚠️ Disclaimer

General health information from NIH sources only. Not a substitute for professional medical advice.

---

## 👤 Author

**Akash Sikarwar** | B.Tech IT, MMMUT Gorakhpur | [@Achiever199](https://github.com/Achiever199)
