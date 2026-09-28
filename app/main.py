from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import db
from .rag import User, authenticate, provider_status, query

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="SentinelRAG", version="2.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)


class DocumentRequest(BaseModel):
    title: str = Field(min_length=2, max_length=200)
    content: str = Field(min_length=10, max_length=20_000)
    allowed_roles: list[str] = Field(min_length=1)


def current_user(x_demo_user: str | None = Header(default=None)) -> User:
    try:
        return authenticate(x_demo_user)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "sentinelrag", **provider_status()}


@app.post("/query")
def run_query(request: QueryRequest, user: User = Depends(current_user)):
    try:
        answer, chunks = query(user, request.question, request.top_k)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "answer": answer,
        "tenant": user.tenant_id,
        "role": user.role,
        "providers": provider_status(),
        "citations": [
            {"document_id": c.document_id, "title": c.title, "score": round(c.score, 4), "text": c.text}
            for c in chunks
        ],
    }


@app.post("/documents", status_code=201)
def add_document(request: DocumentRequest, user: User = Depends(current_user)):
    if user.role != "admin":
        db.write_audit(user.username, user.tenant_id, "denied_ingest", None, [], 0)
        raise HTTPException(status_code=403, detail="Only tenant admins can ingest documents")
    invalid_roles = set(request.allowed_roles) - {"admin", "analyst", "viewer"}
    if invalid_roles:
        raise HTTPException(status_code=422, detail=f"Unknown roles: {sorted(invalid_roles)}")
    document_id = db.create_document(user.tenant_id, request.title, request.content, request.allowed_roles)
    db.write_audit(user.username, user.tenant_id, "ingest", None, [document_id], 1)
    return {"document_id": document_id, "tenant": user.tenant_id}


@app.get("/audit")
def audit(user: User = Depends(current_user)):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Only tenant admins can view audit events")
    return [dict(row) for row in db.list_audit(user.tenant_id)]
