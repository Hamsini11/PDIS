"""
models.py — Pydantic request/response models
FastAPI uses these for automatic validation.
If request doesn't match → 422 error automatically.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


# ─── REQUEST MODELS ──────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    session_id: str
    filter_doc: Optional[str] = None

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        """Block prompt injection attempts."""
        injection_patterns = [
            "ignore previous",
            "ignore all instructions",
            "system prompt",
            "you are now",
            "disregard",
            "forget everything",
            "new instructions",
            "override"
        ]
        q_lower = v.lower()
        if any(p in q_lower for p in injection_patterns):
            raise ValueError("Invalid query detected")
        return v.strip()


class CompareRequest(BaseModel):
    filename_a: str
    filename_b: str


class DocumentFilter(BaseModel):
    filter_doc: Optional[str] = None
    flagged_only: bool = False


# ─── RESPONSE MODELS ─────────────────────────────────────────

class ChatResponse(BaseModel):
    answer: str
    route: str
    session_id: str
    query_type: str


class DocumentInfo(BaseModel):
    filename: str
    indexed_at: Optional[datetime] = None
    chunk_count: Optional[int] = None


class DocumentListResponse(BaseModel):
    documents: list[str]
    total: int


class DateEntry(BaseModel):
    raw_date: str
    normalized: Optional[str]
    page: int
    doc: str
    flagged: bool
    context: Optional[str]


class DateListResponse(BaseModel):
    dates: list[DateEntry]
    total: int
    flagged_count: int


class CompareResponse(BaseModel):
    doc_a: str
    doc_b: str
    summary: str
    fields: list[dict]
    api_calls: int


class UserInfo(BaseModel):
    user_id: str
    email: str
    display_name: str


class ErrorResponse(BaseModel):
    detail: str
    code: str