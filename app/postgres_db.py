from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinelrag")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    role TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    allowed_roles JSONB NOT NULL,
    embedding vector({EMBEDDING_DIMENSIONS}),
    created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_tenant_idx ON documents (tenant_id);
CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING hnsw (embedding vector_cosine_ops);
CREATE TABLE IF NOT EXISTS audit_events (
    id BIGSERIAL PRIMARY KEY,
    username TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    question TEXT,
    document_ids JSONB NOT NULL,
    result_count INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
"""


def utc_now():
    return datetime.now(timezone.utc)


def _connect():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("Install psycopg[binary] to use DB_BACKEND=postgres") from exc
    return psycopg.connect(DATABASE_URL)


@contextmanager
def connection() -> Iterator:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _vector_literal(vector: list[float]) -> str:
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Embedding dimension {len(vector)} does not match EMBEDDING_DIMENSIONS={EMBEDDING_DIMENSIONS}"
        )
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def init_db() -> None:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(SCHEMA)
            users = [
                ("alice", "acme", "admin"),
                ("bob", "acme", "analyst"),
                ("cara", "acme", "viewer"),
                ("diego", "globex", "analyst"),
            ]
            cursor.executemany(
                "INSERT INTO users(username, tenant_id, role) VALUES (%s,%s,%s) ON CONFLICT (username) DO NOTHING",
                users,
            )
            cursor.execute("SELECT COUNT(*) FROM documents")
            if cursor.fetchone()[0] == 0:
                from .embeddings import embed_texts

                seed = [
                    ("acme", "Acme refund policy", "Acme customers can request a refund within 30 days of purchase. Enterprise contracts may extend this window to 60 days.", ["admin", "analyst", "viewer"]),
                    ("acme", "Acme internal roadmap", "The Q4 roadmap prioritizes offline mode, SSO, and lower sync latency for enterprise teams.", ["admin", "analyst"]),
                    ("acme", "Acme finance controls", "Only finance administrators may approve invoice adjustments above 10,000 dollars.", ["admin"]),
                    ("globex", "Globex refund policy", "Globex customers can request a refund within 14 days of purchase when the service has not been materially used.", ["admin", "analyst", "viewer"]),
                ]
                vectors = embed_texts([content for _, _, content, _ in seed])
                cursor.executemany(
                    "INSERT INTO documents(tenant_id,title,content,allowed_roles,embedding,created_at) VALUES (%s,%s,%s,%s,%s::vector,%s)",
                    [
                        (tenant, title, content, json.dumps(roles), _vector_literal(vector), utc_now())
                        for (tenant, title, content, roles), vector in zip(seed, vectors)
                    ],
                )


def get_user(username: str):
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT username, tenant_id, role FROM users WHERE username = %s", (username,))
            row = cursor.fetchone()
            if row is None:
                return None
            return {"username": row[0], "tenant_id": row[1], "role": row[2]}


def create_document(tenant_id: str, title: str, content: str, allowed_roles: list[str]) -> int:
    from .embeddings import embed_texts

    embedding = _vector_literal(embed_texts([content])[0])
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO documents(tenant_id,title,content,allowed_roles,embedding,created_at) VALUES (%s,%s,%s,%s,%s::vector,%s) RETURNING id",
                (tenant_id, title, content, json.dumps(sorted(set(allowed_roles))), embedding, utc_now()),
            )
            return int(cursor.fetchone()[0])


def list_documents_for_user(tenant_id: str, role: str):
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, tenant_id, title, content, allowed_roles, created_at FROM documents WHERE tenant_id = %s AND allowed_roles ? %s ORDER BY id",
                (tenant_id, role),
            )
            columns = [description.name for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


def search_authorized_chunks(tenant_id: str, role: str, question: str, top_k: int):
    from .embeddings import embed_texts

    query_embedding = _vector_literal(embed_texts([question])[0])
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, title, content, 1 - (embedding <=> %s::vector) AS score
                FROM documents
                WHERE tenant_id = %s AND allowed_roles ? %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, tenant_id, role, query_embedding, top_k),
            )
            return [(row[0], row[1], row[2], float(row[3])) for row in cursor.fetchall()]


def write_audit(username: str, tenant_id: str, event_type: str, question: str | None, document_ids: list[int], result_count: int) -> None:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO audit_events(username,tenant_id,event_type,question,document_ids,result_count,created_at) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (username, tenant_id, event_type, question, json.dumps(document_ids), result_count, utc_now()),
            )


def list_audit(tenant_id: str, limit: int = 50):
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, username, tenant_id, event_type, question, document_ids, result_count, created_at FROM audit_events WHERE tenant_id = %s ORDER BY id DESC LIMIT %s",
                (tenant_id, limit),
            )
            columns = [description.name for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


def backend_status() -> dict[str, str]:
    return {"database_backend": "postgres", "vector_backend": "pgvector cosine distance"}
