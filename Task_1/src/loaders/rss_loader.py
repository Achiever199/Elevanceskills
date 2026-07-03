from __future__ import annotations

import logging
import feedparser

logger = logging.getLogger("kb_system")


class RSSLoader:
    """Pulls entries from one or more RSS/Atom feeds."""

    def __init__(self, source_id: str, options: dict):
        self.source_id = source_id
        self.feed_urls = options.get("feed_urls", [])

    def load(self):
        from . import RawDocument

        docs: list[RawDocument] = []
        for feed_url in self.feed_urls:
            try:
                parsed = feedparser.parse(feed_url)
            except Exception as exc:  # noqa: BLE001
                logger.error("[%s] failed to parse feed %s: %s", self.source_id, feed_url, exc)
                continue

            for entry in parsed.entries:
                entry_id = entry.get("id") or entry.get("link")
                title = entry.get("title", "")
                body = entry.get("summary", "") or entry.get("description", "")
                text = f"{title}\n\n{body}".strip()
                if not text:
                    continue
                docs.append(
                    RawDocument(
                        doc_id=entry_id or f"{feed_url}#{title}",
                        text=text,
                        metadata={
                            "source_type": "rss",
                            "source_id": self.source_id,
                            "feed_url": feed_url,
                            "title": title,
                            "link": entry.get("link", ""),
                        },
                    )
                )
        logger.info("[%s] loaded %d entr(y/ies)", self.source_id, len(docs))
        return docs
