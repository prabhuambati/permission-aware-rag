from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

DB_PATH = Path(os.getenv("SENTINELRAG_DB", "sentinelrag.sqlite3"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    role TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    allowed_roles TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    question TEXT,
    document_ids TEXT NOT NULL,
    result_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connection() as conn:
        conn.executescript(SCHEMA)
        users = [
            ("alice", "acme", "admin"),
            ("bob", "acme", "analyst"),
            ("cara", "acme", "viewer"),
            ("diego", "globex", "analyst"),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO users(username, tenant_id, role) VALUES (?, ?, ?)", users
        )
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if count == 0:
            seed = [
                ("acme", "Acme refund policy", "Acme customers can request a refund within 30 days of purchase. Enterprise contracts may extend this window to 60 days.", ["admin", "analyst", "viewer"]),
                ("acme", "Acme internal roadmap", "The Q4 roadmap prioritizes offline mode, SSO, and lower sync latency for enterprise teams.", ["admin", "analyst"]),
                ("acme", "Acme finance controls", "Only finance administrators may approve invoice adjustments above 10,000 dollars.", ["admin"]),
                ("globex", "Globex refund policy", "Globex customers can request a refund within 14 days of purchase when the service has not been materially used.", ["admin", "analyst", "viewer"]),
            ]
            conn.executemany(
                "INSERT INTO documents(tenant_id,title,content,allowed_roles,created_at) VALUES (?,?,?,?,?)",
                [(t, title, content, json.dumps(roles), utc_now()) for t, title, content, roles in seed],
            )


def get_user(username: str) -> sqlite3.Row | None:
    with connection() as conn:
        return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()


def create_document(tenant_id: str, title: str, content: str, allowed_roles: list[str]) -> int:
    with connection() as conn:
        cursor = conn.execute(
            "INSERT INTO documents(tenant_id,title,content,allowed_roles,created_at) VALUES (?,?,?,?,?)",
            (tenant_id, title, content, json.dumps(sorted(set(allowed_roles))), utc_now()),
        )
        return int(cursor.lastrowid)


def list_documents_for_user(tenant_id: str, role: str) -> list[sqlite3.Row]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE tenant_id = ? ORDER BY id",
            (tenant_id,),
        ).fetchall()
    return [row for row in rows if role in json.loads(row["allowed_roles"])]


def write_audit(username: str, tenant_id: str, event_type: str, question: str | None, document_ids: list[int], result_count: int) -> None:
    with connection() as conn:
        conn.execute(
            "INSERT INTO audit_events(username,tenant_id,event_type,question,document_ids,result_count,created_at) VALUES (?,?,?,?,?,?,?)",
            (username, tenant_id, event_type, question, json.dumps(document_ids), result_count, utc_now()),
        )


def list_audit(tenant_id: str, limit: int = 50) -> list[sqlite3.Row]:
    with connection() as conn:
        return conn.execute(
            "SELECT * FROM audit_events WHERE tenant_id = ? ORDER BY id DESC LIMIT ?",
            (tenant_id, limit),
        ).fetchall()


def search_authorized_chunks(tenant_id: str, role: str, question: str, top_k: int):
    raise NotImplementedError("SQLite uses application-level embeddings; use DB_BACKEND=postgres for pgvector search")


def backend_status() -> dict[str, str]:
    return {"database_backend": "sqlite", "vector_backend": "application cosine similarity"}
