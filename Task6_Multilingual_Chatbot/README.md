# 🌍 Multilingual Conversational Chatbot

> **Internship Project — Elevance skills**
> Task 6 | Built by [Akash Sikarwar](https://github.com/Achiever199)
> Live Preview: **https://task6multilingualchatbot-vjadnwqty5txfpysrmavni.streamlit.app/**

---

## 📌 Problem Statement

Build a chatbot that supports multilingual conversations across at least 3 additional languages while preserving context, intent, and conversational continuity throughout language switches.

---

## 🌐 Supported Languages (10 Total)

| Language | Code | Flag |
|---|---|---|
| English | en | 🇬🇧 |
| Hindi | hi | 🇮🇳 |
| French | fr | 🇫🇷 |
| Spanish | es | 🇪🇸 |
| German | de | 🇩🇪 |
| Chinese | zh-cn | 🇨🇳 |
| Arabic | ar | 🇸🇦 |
| Portuguese | pt | 🇧🇷 |
| Japanese | ja | 🇯🇵 |
| Russian | ru | 🇷🇺 |

---

## 🗂️ Project Structure

```
Task6_Multilingual_Chatbot/
├── app.py                  # Streamlit UI
├── language_detector.py    # langdetect-based language identification
├── translator.py           # deep-translator (Google backend, free)
├── intent_engine.py        # Intent detection + context management
├── requirements.txt
└── README.md
```

---

## 🧠 Architecture

```
User Input (Any Language)
        ↓
Language Detection (langdetect)
        ↓
Translate to English (deep-translator)
        ↓
Intent Detection (keyword matching)
        ↓
Generate English Response
        ↓
Translate Back to User's Language
        ↓
Context Update (track lang switches, topics)
        ↓
Display Response with Language Badge
```

---

## ✅ Features

- **Auto language detection** with confidence score
- **10 supported languages** including right-to-left (Arabic)
- **Language switch detection** with acknowledgement banner
- **Context retention** across turns and language switches
- **Mixed language input** handling
- **Live analytics sidebar** — language distribution pie chart + timeline
- **12 example prompts** across all languages

---

## 🚀 Setup & Run

```bash
# 1. Create folder
mkdir Task6_Multilingual_Chatbot && cd Task6_Multilingual_Chatbot

# 2. Virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install (all lightweight, no torch needed!)
pip install streamlit langdetect deep-translator pandas plotly

# 4. Run
streamlit run app.py
```

---

## 📊 Evaluation Criteria

| Criterion | Implementation |
|---|---|
| Language identification | langdetect with confidence score shown per message |
| 3+ additional languages | 10 languages supported (9 non-English) |
| Context retention | ConversationContext dataclass tracks turns, topics, lang history |
| Language switch handling | Switch banner + acknowledgement in target language |
| Mixed language input | is_mixed_language() splits and detects per segment |
| Cross-lingual reasoning | Translate → English intent → translate back pipeline |

---

## 👤 Author

**Akash Sikarwar** | B.Tech IT, MMMUT Gorakhpur | [@Achiever199](https://github.com/Achiever199)
