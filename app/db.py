from __future__ import annotations

import os

BACKEND = os.getenv("DB_BACKEND", "sqlite").lower()
if BACKEND == "postgres":
    from . import postgres_db as _impl
elif BACKEND == "sqlite":
    from . import sqlite_db as _impl
else:
    raise ValueError(f"Unsupported DB_BACKEND: {BACKEND}")


def init_db():
    return _impl.init_db()


def get_user(username: str):
    return _impl.get_user(username)


def create_document(tenant_id: str, title: str, content: str, allowed_roles: list[str]):
    return _impl.create_document(tenant_id, title, content, allowed_roles)


def list_documents_for_user(tenant_id: str, role: str):
    return _impl.list_documents_for_user(tenant_id, role)


def search_authorized_chunks(tenant_id: str, role: str, question: str, top_k: int):
    return _impl.search_authorized_chunks(tenant_id, role, question, top_k)


def write_audit(username: str, tenant_id: str, event_type: str, question: str | None, document_ids: list[int], result_count: int):
    return _impl.write_audit(username, tenant_id, event_type, question, document_ids, result_count)


def list_audit(tenant_id: str, limit: int = 50):
    return _impl.list_audit(tenant_id, limit)


def backend_status() -> dict[str, str]:
    return _impl.backend_status()
