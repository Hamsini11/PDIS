"""
routes/chat.py — Chat endpoint
POST /api/chat
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from backend.models import ChatRequest, ChatResponse
from backend.auth import get_demo_user
from backend.security import limiter, log_audit
from supabase_client import save_message, load_chat_history

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from router import classify_query
from vector_store import search, get_all_dates, list_documents
from anthropic import Anthropic

import os
print(f"DEBUG CWD: {os.getcwd()}")
print(f"DEBUG index exists: {os.path.exists('storage/faiss.index')}")
from vector_store import list_documents
print(f"DEBUG docs: {list_documents()}")

router = APIRouter()
client = Anthropic()


def handle_structured(query: str, filter_doc: str = None) -> str:
    dates = get_all_dates(filter_doc=filter_doc)
    if not dates:
        return "No dates found in the indexed documents."

    date_lines = []
    for d in dates:
        flag = "⚠️ FLAGGED" if d.get("flagged") else "✅"
        date_lines.append(
            f"{flag} | {d['raw_date']} → {d.get('normalized','N/A')} "
            f"| Page {d['page']} | {d['doc'][:40]}"
        )

    prompt = f"""You are a pharmaceutical document compliance assistant.
Answer using ONLY the date records below. Never guess.

DATE RECORDS:
{chr(10).join(date_lines[:80])}

QUESTION: {query}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def handle_semantic(query: str, filter_doc: str = None) -> str:
    filter_arg = None if filter_doc == "All documents" else filter_doc
    k = 15 if filter_arg else 10
    chunks = search(query, top_k=k, filter_doc=filter_arg, use_hyde=True)

    if not chunks:
        return "No relevant content found in the indexed documents."

    context = "\n\n---\n\n".join([
        f"[{c['doc']} | Page {c['page']}]\n{c['text']}"
        for c in chunks
    ])

    prompt = f"""You are a pharmaceutical document compliance assistant.
Answer using ONLY the context below.
Always cite document name and page number.
If not in context: "This information was not found in the indexed documents."
Never guess or hallucinate.

CONTEXT:
{context}

QUESTION: {query}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def handle_listing() -> str:
    docs = list_documents()
    if not docs:
        return "No documents indexed yet."
    return "Indexed documents:\n" + "\n".join(f"- {d}" for d in docs)


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(
        request_data: ChatRequest,
        request: Request,
        user: dict = Depends(get_demo_user)):
    """
    Main chat endpoint.
    Classifies query → routes to correct handler → saves to history.
    """
    try:
        # Classify query intent
        classification = classify_query(request_data.query)
        query_type = classification["query_type"]
        filter_doc = request_data.filter_doc

        # Route to correct handler
        if query_type == "listing":
            route = "📋 Document Listing"
            answer = handle_listing()
        elif query_type == "date_extraction":
            route = "📋 Date Extraction"
            answer = handle_structured(request_data.query, filter_doc)
        else:
            route = "🔍 Semantic (RAG)"
            answer = handle_semantic(request_data.query, filter_doc)

        # Save to chat history
        try:
            save_message(user["ms_user_id"],
                        request_data.session_id,
                        "user", request_data.query)
            save_message(user["ms_user_id"],
                        request_data.session_id,
                        "assistant", answer)
        except Exception as e:
            print(f"  Chat history save failed: {e}")

        # Audit log
        log_audit(
            user_id=user["ms_user_id"],
            action="QUERY",
            resource="chat",
            details=f"type={query_type}"
        )

        return ChatResponse(
            answer=answer,
            route=route,
            session_id=request_data.session_id,
            query_type=query_type
        )

    except ValueError as e:
        # Prompt injection caught by Pydantic validator
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))