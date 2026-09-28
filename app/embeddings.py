from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from typing import Protocol

TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


class EmbeddingProvider(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalHashEmbeddings:
    """Small deterministic fallback used for offline demos and tests.

    This is not intended to replace a semantic model; it only keeps the project runnable
    when no embedding API key is available.
    """

    name = "local-hash"
    dimensions = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in TOKEN_RE.findall(text.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] % 2 else -1.0
                vector[index] += sign
            vectors.append(_normalize(vector))
        return vectors


class OpenAIEmbeddings:
    name = "openai"

    def __init__(self, model: str):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package to use OPENAI embeddings") from exc
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [_normalize(item.embedding) for item in response.data]


_EMBEDDING_CACHE: dict[tuple[str, str, str], list[float]] = {}


def _build_provider(provider_name: str, model: str) -> EmbeddingProvider:
    if provider_name == "openai":
        return OpenAIEmbeddings(model)
    if provider_name in {"local", "local-hash"}:
        return LocalHashEmbeddings()
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider_name}")


def get_provider() -> EmbeddingProvider:
    provider_name = os.getenv("EMBEDDING_PROVIDER", "local-hash").lower()
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    return _build_provider(provider_name, model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    provider_name = os.getenv("EMBEDDING_PROVIDER", "local-hash").lower()
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    provider = get_provider()
    if provider_name != "openai":
        return provider.embed(texts)

    keys = [(provider_name, model, text) for text in texts]
    missing_texts = list(dict.fromkeys(text for key, text in zip(keys, texts) if key not in _EMBEDDING_CACHE))
    if missing_texts:
        # One network request for the query and all uncached authorized chunks.
        vectors = provider.embed(missing_texts)
        for text, vector in zip(missing_texts, vectors):
            _EMBEDDING_CACHE[(provider_name, model, text)] = vector
    return [_EMBEDDING_CACHE[key] for key in keys]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector
