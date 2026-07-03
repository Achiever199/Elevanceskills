"""Loads and validates the YAML configuration file."""
from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppConfig:
    raw: dict[str, Any]

    @property
    def vector_store(self) -> dict:
        return self.raw.get("vector_store", {})

    @property
    def chunking(self) -> dict:
        return self.raw.get("chunking", {"chunk_size": 1000, "chunk_overlap": 150})

    @property
    def manifest_path(self) -> str:
        return self.raw.get("manifest", {}).get("path", "./data/manifest.sqlite3")

    @property
    def logging_cfg(self) -> dict:
        return self.raw.get("logging", {"level": "INFO", "file": "./logs/kb_system.log"})

    @property
    def sources(self) -> list[dict]:
        return [s for s in self.raw.get("sources", []) if s.get("enabled", True)]


def load_config(path: str = "config.yaml") -> AppConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig(raw=raw)
