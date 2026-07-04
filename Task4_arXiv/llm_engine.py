"""
llm_engine.py
-------------
Open-source LLM backend (Hugging Face Transformers) used for:
  - abstractive summarization of paper abstracts
  - explanation generation / question answering grounded in retrieved
    paper excerpts (the "generation" half of RAG)

Default model is google/flan-t5-base: small enough to run on a CPU-only
laptop, and instruction-tuned so a single model can do both summarization
and explanation via different prompt templates (no need to load two
separate models). A larger model (flan-t5-large) or a chat model
(microsoft/Phi-3-mini-4k-instruct) can be swapped in via MODEL_NAME if the
machine has more RAM/a GPU.

Models are downloaded from the Hugging Face Hub the first time they're
used and cached locally afterwards (~/.cache/huggingface).
"""

from functools import lru_cache
from typing import Optional

DEFAULT_MODEL = "google/flan-t5-base"


@lru_cache(maxsize=2)
def _load_pipeline(model_name: str):
    """Lazily load and cache a text2text-generation pipeline. Cached so the
    (slow, one-time) model load only happens once per process even though
    Streamlit reruns the script on every interaction."""
    from transformers import pipeline

    return pipeline("text2text-generation", model=model_name)


class LLMEngine:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name

    @property
    def pipe(self):
        return _load_pipeline(self.model_name)

    def summarize(self, text: str, max_new_tokens: int = 120) -> str:
        """Abstractive summary of a paper abstract (or any passage)."""
        text = (text or "").strip()
        if not text:
            return ""
        prompt = f"Summarize the following research paper abstract in 2-3 sentences:\n\n{text}"
        output = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return output[0]["generated_text"].strip()

    def explain(
        self,
        question: str,
        context: str,
        history: Optional[str] = None,
        max_new_tokens: int = 200,
    ) -> str:
        """Grounded explanation/QA generation. `context` is the concatenated
        excerpts retrieved from relevant papers (the RAG evidence)."""
        history_block = f"Earlier in the conversation:\n{history}\n\n" if history else ""
        prompt = (
            "You are an expert computer science research assistant. Answer the "
            "question using ONLY the information in the context below. If the "
            "context does not contain enough information, say so explicitly "
            "rather than guessing.\n\n"
            f"{history_block}"
            f"Context (excerpts from research papers):\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        output = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return output[0]["generated_text"].strip()

    def explain_concept(self, concept: str, context: str = "", max_new_tokens: int = 200) -> str:
        """Standalone concept explanation (e.g. 'explain attention mechanisms'),
        optionally grounded in retrieved context if available."""
        context_block = f"\n\nRelevant excerpts from papers:\n{context}" if context else ""
        prompt = (
            "You are an expert computer science tutor. Explain the following "
            "concept clearly, at a level suitable for a graduate student who is "
            f"new to the topic.{context_block}\n\nConcept: {concept}\nExplanation:"
        )
        output = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return output[0]["generated_text"].strip()
