# 🤖 Multimodal AI Assistant — Text + Vision (Gemini Edition)
> **Internship Project — inAmigos Foundation | AI Data Analyst Intern**
> Task 2 | Built by [Akash Sikarwar](https://github.com/Achiever199)

This is a drop-in replacement for the original Anthropic-powered assistant.
It uses **Google Gemini's free API tier** instead, so no Anthropic API key
(and no cost) is required.

---
## 📌 Problem Statement
Develop a multi-modal AI assistant capable of understanding and reasoning over
both text and image inputs. The assistant should analyze visual content,
extract relevant information, maintain conversational context, and generate
evidence-based responses with ambiguity handling and response validation.

---
## 🗂️ Project Structure
```
Task2_Multimodal_Assistant_Gemini/
├── app.py                  # Streamlit UI + main pipeline
├── vision_analyzer.py      # Image encoding, prompt engineering, response validation
├── context_manager.py      # Multi-turn context, intent detection, prompt-context building
├── requirements.txt
└── README.md
```

---
## 🧠 Architecture
```
User Input (Text + Optional Image)
         ↓
Image Resize/Encode                  Intent Detection
         ↓                                  ↓
Build Analysis Prompt ←── Conversation History Summary
         ↓
Google Gemini Vision API (gemini-2.5-flash)
         ↓
Response Validation (5 quality checks)
         ↓
Context Update (store turn, image note, intent, quality)
         ↓
Display with Quality Badge + Validation Toggle
```

---
## ✅ Features
| Feature | Implementation |
|---|---|
| **Visual Analysis** | Gemini Vision (`gemini-2.5-flash`) |
| **Text + Image** | Unified multimodal input handling |
| **Context Retention** | Conversation history summary injected into every prompt |
| **Intent Detection** | 9 intents: describe, identify, extract, compare, analyze, count, locate, sentiment, general |
| **Ambiguity Handling** | System prompt enforces confidence flags ("appears to", "I'm not certain") |
| **Response Validation** | 5-check quality scoring: observations, reasoning, uncertainty, substance, no hallucination |
| **Multi-image Support** | Previous images referenced by short description in context |
| **Quality Dashboard** | Per-response score + trend chart in sidebar |

---
## 🚀 Setup & Run
```bash
# 1. Create folder
mkdir Task2_Multimodal_Assistant_Gemini && cd Task2_Multimodal_Assistant_Gemini
# (copy app.py, vision_analyzer.py, context_manager.py, requirements.txt here)

# 2. Virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies (all lightweight!)
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

Get your **free** Gemini API key at: https://aistudio.google.com/apikey
(No credit card required. Paste it into the sidebar field when the app opens —
it's kept only in the browser session, never written to disk.)

### Which model to pick?
- **gemini-2.5-flash** — best all-round choice, generous free-tier limits. Default.
- **gemini-2.5-flash-lite** — fastest/cheapest, good for simple description/extraction tasks.
- **gemini-2.5-pro** — strongest reasoning, but a much lower free daily request cap.

---
## 📊 Evaluation Criteria Coverage
| Criterion | How Addressed |
|---|---|
| Visual content analysis | Gemini Vision analyzes every attached image |
| Extract relevant information | Structured prompt extracts: objects, text, colors, context |
| Maintain conversational context | `ConversationContext` class + rolling history summary in every prompt |
| Evidence-based responses | System prompt enforces grounding ("I can see...", "the image shows...") |
| Contextual reasoning | History summary + past-image notes injected into every prompt |
| Ambiguity handling | Confidence flags enforced in system prompt + validated in response |
| Response validation | 5-check validator scores every response 0–100% |
| Intelligent decision-making | Intent detection routes prompt style per query |

---
## ⚠️ Notes on the Gemini free tier
- Free tier currently covers the **Flash** and **Flash-Lite** model families;
  Pro models have a much smaller free daily request quota.
- Rate limits (requests/minute and /day) are enforced per Google Cloud project,
  not per API key — see https://ai.google.dev/gemini-api/docs/rate-limits for
  current numbers, since Google updates these periodically.
- If you see a `429` error in the app, you've hit the free-tier rate limit —
  wait a minute and retry, or switch to `gemini-2.5-flash-lite`.

---
## 👤 Author
**Akash Sikarwar** | B.Tech IT, MMMUT Gorakhpur | [@Achiever199](https://github.com/Achiever199)
