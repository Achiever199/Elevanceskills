from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawDocument:
    """A single unit of content pulled from a source, pre-chunking."""
    doc_id: str            # stable unique id within its source (e.g. file path, URL, RSS entry id)
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLoader:
    """Interface every source loader must implement."""

    def __init__(self, source_id: str, options: dict):
        self.source_id = source_id
        self.options = options

    def load(self) -> list[RawDocument]:
        raise NotImplementedError


from .file_loader import FileLoader   # noqa: E402
from .web_loader import WebLoader     # noqa: E402
from .rss_loader import RSSLoader     # noqa: E402

LOADER_REGISTRY = {
    "file": FileLoader,
    "web": WebLoader,
    "rss": RSSLoader,
}


def get_loader(source_cfg: dict) -> BaseLoader:
    loader_cls = LOADER_REGISTRY.get(source_cfg["type"])
    if loader_cls is None:
        raise ValueError(f"Unknown source type: {source_cfg['type']}")
    return loader_cls(source_id=source_cfg["id"], options=source_cfg.get("options", {}))
