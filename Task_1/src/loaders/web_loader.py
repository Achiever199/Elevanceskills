from __future__ import annotations

import logging
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger("kb_system")

HEADERS = {"User-Agent": "KnowledgeBaseBot/1.0 (+https://example.com/bot)"}


class WebLoader:
    """Fetches and extracts readable text from a list of URLs."""

    def __init__(self, source_id: str, options: dict):
        self.source_id = source_id
        self.urls = options.get("urls", [])
        self.timeout = options.get("timeout_seconds", 15)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _fetch(self, url: str) -> str:
        resp = requests.get(url, headers=HEADERS, timeout=self.timeout)
        resp.raise_for_status()
        return resp.text

    @staticmethod
    def _extract_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [ln.strip() for ln in text.splitlines()]
        return "\n".join(ln for ln in lines if ln)

    def load(self):
        from . import RawDocument

        docs: list[RawDocument] = []
        for url in self.urls:
            try:
                html = self._fetch(url)
                text = self._extract_text(html)
            except Exception as exc:  # noqa: BLE001
                logger.error("[%s] failed to fetch %s: %s", self.source_id, url, exc)
                continue

            if not text.strip():
                continue

            docs.append(
                RawDocument(
                    doc_id=url,
                    text=text,
                    metadata={"source_type": "web", "source_id": self.source_id, "url": url},
                )
            )
        logger.info("[%s] loaded %d page(s)", self.source_id, len(docs))
        return docs
