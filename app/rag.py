from __future__ import annotations

import os

from . import db
from .embeddings import cosine_similarity, embed_texts
from .llm import get_generator
from .rag_types import Chunk

MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.18"))


class User:
    def __init__(self, username: str, tenant_id: str, role: str):
        self.username = username
        self.tenant_id = tenant_id
        self.role = role


def authenticate(username: str | None) -> User:
    if not username:
        raise PermissionError("Missing X-Demo-User header")
    record = db.get_user(username)
    if record is None:
        raise PermissionError("Unknown demo user")
    return User(record["username"], record["tenant_id"], record["role"])


def _chunks(content: str, size: int = 70) -> list[str]:
    words = content.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)] or [content]


def retrieve(user: User, question: str, top_k: int = 4) -> list[Chunk]:
    # SECURITY INVARIANT: tenant and role policy filtering happens before embedding/ranking.
    if db.BACKEND == "postgres":
        rows = db.search_authorized_chunks(user.tenant_id, user.role, question, max(top_k * 3, 12))
        return [
            Chunk(document_id, title, text, score)
            for document_id, title, text, score in rows
            if score >= MIN_RELEVANCE
        ][:top_k]

    allowed_documents = db.list_documents_for_user(user.tenant_id, user.role)
    candidates: list[tuple[int, str, str]] = []
    for document in allowed_documents:
        for text in _chunks(document["content"]):
            candidates.append((document["id"], document["title"], text))

    if not candidates:
        return []

    vectors = embed_texts([question] + [text for _, _, text in candidates])
    query_vector = vectors[0]
    ranked: list[Chunk] = []
    for (document_id, title, text), vector in zip(candidates, vectors[1:]):
        score = cosine_similarity(query_vector, vector)
        if score >= MIN_RELEVANCE:
            ranked.append(Chunk(document_id, title, text, score))
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:top_k]


def query(user: User, question: str, top_k: int = 4) -> tuple[str, list[Chunk]]:
    chunks = retrieve(user, question, top_k)
    answer = get_generator().generate(question, chunks)
    db.write_audit(
        user.username,
        user.tenant_id,
        "query",
        question,
        sorted({chunk.document_id for chunk in chunks}),
        len(chunks),
    )
    return answer, chunks


def provider_status() -> dict[str, str]:
    return {
        **db.backend_status(),
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "local-hash"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "llm_provider": os.getenv("LLM_PROVIDER", "extractive"),
        "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
    }
