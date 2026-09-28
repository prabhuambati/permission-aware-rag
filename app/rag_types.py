from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    document_id: int
    title: str
    text: str
    score: float
