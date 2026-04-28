import os
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime
import uuid

load_dotenv()

# ─── CLIENT ──────────────────────────────────────────────────
url  = os.environ.get("SUPABASE_URL")
key  = os.environ.get("SUPABASE_ANON_KEY")

supabase: Client = create_client(url, key)


# ─── USER HELPERS ────────────────────────────────────────────
def get_or_create_user(ms_user_id: str, email: str = None,
                        display_name: str = None) -> dict:
    """
    Get existing user or create new one on first login.
    Returns user dict with id field.
    """
    # check if user exists
    result = supabase.table("users")\
        .select("*")\
        .eq("ms_user_id", ms_user_id)\
        .execute()

    if result.data:
        return result.data[0]

    # create new user
    new_user = {
        "ms_user_id":    ms_user_id,
        "email":         email,
        "display_name":  display_name
    }
    result = supabase.table("users").insert(new_user).execute()
    return result.data[0]


# ─── CHAT HISTORY HELPERS ────────────────────────────────────
def save_message(user_id: str, session_id: str,
                  role: str, content: str):
    """Save a single chat message to Supabase."""
    supabase.table("chat_history").insert({
        "user_id":    user_id,
        "session_id": session_id,
        "role":       role,
        "content":    content
    }).execute()


def load_chat_history(user_id: str,
                       session_id: str) -> list[dict]:
    """
    Load chat history for a user session.
    Returns list of {role, content} dicts.
    """
    result = supabase.table("chat_history")\
        .select("role, content, created_at")\
        .eq("user_id", user_id)\
        .eq("session_id", session_id)\
        .order("created_at")\
        .execute()

    return [
        {"role": r["role"], "content": r["content"]}
        for r in result.data
    ]


def get_user_sessions(user_id: str) -> list[str]:
    """Get all session IDs for a user."""
    result = supabase.table("chat_history")\
        .select("session_id")\
        .eq("user_id", user_id)\
        .execute()

    seen = set()
    sessions = []
    for r in result.data:
        if r["session_id"] not in seen:
            seen.add(r["session_id"])
            sessions.append(r["session_id"])
    return sessions


# ─── VENDOR HELPERS ──────────────────────────────────────────
def save_vendor(user_id: str, vendor_name: str,
                 compliance_email: str) -> dict:
    """Save or update vendor email."""
    # check if vendor exists for this user
    result = supabase.table("vendors")\
        .select("*")\
        .eq("user_id", user_id)\
        .eq("vendor_name", vendor_name)\
        .execute()

    if result.data:
        # update existing
        supabase.table("vendors")\
            .update({"compliance_email": compliance_email})\
            .eq("id", result.data[0]["id"])\
            .execute()
        return result.data[0]

    # insert new
    result = supabase.table("vendors").insert({
        "user_id":          user_id,
        "vendor_name":      vendor_name,
        "compliance_email": compliance_email
    }).execute()
    return result.data[0]


def get_vendor_email(user_id: str,
                      vendor_name: str) -> str | None:
    """Get compliance email for a vendor."""
    result = supabase.table("vendors")\
        .select("compliance_email")\
        .eq("user_id", user_id)\
        .eq("vendor_name", vendor_name)\
        .execute()

    if result.data:
        return result.data[0]["compliance_email"]
    return None


# ─── DOCUMENT HELPERS ────────────────────────────────────────
def save_document(user_id: str, filename: str,
                   packet_summary: str = None,
                   flagged_count: int = 0) -> dict:
    """Save document metadata after indexing."""
    # check if already saved
    result = supabase.table("documents")\
        .select("*")\
        .eq("user_id", user_id)\
        .eq("filename", filename)\
        .execute()

    if result.data:
        return result.data[0]

    result = supabase.table("documents").insert({
        "user_id":        user_id,
        "filename":       filename,
        "packet_summary": packet_summary,
        "flagged_count":  flagged_count
    }).execute()
    return result.data[0]


def get_user_documents(user_id: str) -> list[dict]:
    """Get all documents uploaded by a user."""
    result = supabase.table("documents")\
        .select("*")\
        .eq("user_id", user_id)\
        .order("uploaded_at", desc=True)\
        .execute()
    return result.data


# ─── SESSION ID GENERATOR ─────────────────────────────────────
def new_session_id() -> str:
    """Generate a unique session ID."""
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

# ─── MICROSOFT SIGNIN ─────────────────────────────────────
def sign_in_with_microsoft():
    """Get Microsoft OAuth URL for redirect."""
    result = supabase.auth.sign_in_with_oauth({
        "provider": "azure",
        "options": {
            "redirect_to": os.environ.get("SUPABASE_REDIRECT_URL")
        }
    })
    return result.url