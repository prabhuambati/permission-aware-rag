import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

DB_FILE = Path(__file__).parent / "test.sqlite3"
os.environ["DB_BACKEND"] = "sqlite"
os.environ["SENTINELRAG_DB"] = str(DB_FILE)

from app.main import app  # noqa: E402
from app import db  # noqa: E402


@pytest.fixture(autouse=True)
def reset_db():
    if DB_FILE.exists():
        DB_FILE.unlink()
    db.init_db()
    yield
    if DB_FILE.exists():
        DB_FILE.unlink()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_viewer_cannot_retrieve_admin_only_document(client):
    response = client.post(
        "/query",
        headers={"X-Demo-User": "cara"},
        json={"question": "Who can approve invoice adjustments?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == []
    assert "authorized" in body["answer"]


def test_analyst_can_retrieve_analyst_document_but_not_admin_document(client):
    roadmap = client.post(
        "/query",
        headers={"X-Demo-User": "bob"},
        json={"question": "What is on the Q4 roadmap?"},
    ).json()
    assert any("roadmap" in item["title"].lower() for item in roadmap["citations"])

    finance = client.post(
        "/query",
        headers={"X-Demo-User": "bob"},
        json={"question": "Who can approve invoice adjustments?"},
    ).json()
    assert finance["citations"] == []


def test_tenant_isolation_prevents_cross_tenant_leakage(client):
    response = client.post(
        "/query",
        headers={"X-Demo-User": "diego"},
        json={"question": "What is the Acme Q4 roadmap?"},
    )
    assert response.status_code == 200
    assert all("Acme" not in item["title"] for item in response.json()["citations"])


def test_only_admin_can_ingest_and_audit_is_written(client):
    denied = client.post(
        "/documents",
        headers={"X-Demo-User": "bob"},
        json={"title": "Attempt", "content": "This should not be added.", "allowed_roles": ["analyst"]},
    )
    assert denied.status_code == 403

    created = client.post(
        "/documents",
        headers={"X-Demo-User": "alice"},
        json={"title": "Security notice", "content": "Acme rotates keys every 90 days.", "allowed_roles": ["analyst"]},
    )
    assert created.status_code == 201

    audit = client.get("/audit", headers={"X-Demo-User": "alice"})
    assert audit.status_code == 200
    assert any(event["event_type"] == "ingest" for event in audit.json())
    assert any(event["event_type"] == "denied_ingest" for event in audit.json())
