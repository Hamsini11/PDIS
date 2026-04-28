"""
auth.py — Microsoft OAuth via MSAL
This is why we moved to FastAPI — proper OAuth callback handling.
Streamlit couldn't handle the redirect flow. FastAPI can.
"""

import os
import msal
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer
from jose import jwt, JWTError
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ──────────────────────────────────────────────────
AZURE_CLIENT_ID     = os.environ.get("AZURE_CLIENT_ID")
AZURE_CLIENT_SECRET = os.environ.get("AZURE_CLIENT_SECRET")
AZURE_TENANT_ID     = os.environ.get("AZURE_TENANT_ID", "common")
REDIRECT_URI        = os.environ.get("REDIRECT_URI",
                                      "http://localhost:8000/auth/callback")

AUTHORITY = f"https://login.microsoftonline.com/{AZURE_TENANT_ID}"
SCOPES    = ["openid", "email", "profile", "User.Read"]

security = HTTPBearer(auto_error=False)


# ─── MSAL CLIENT ─────────────────────────────────────────────
def get_msal_app() -> msal.ConfidentialClientApplication:
    """Create MSAL app for OAuth flow."""
    return msal.ConfidentialClientApplication(
        client_id=AZURE_CLIENT_ID,
        client_credential=AZURE_CLIENT_SECRET,
        authority=AUTHORITY
    )


# ─── AUTH URL ────────────────────────────────────────────────
def get_auth_url(state: str = "") -> str:
    """
    Generate Microsoft login URL.
    User visits this URL → logs in → redirected back to /auth/callback
    """
    msal_app = get_msal_app()
    auth_url = msal_app.get_authorization_request_url(
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
        state=state
    )
    return auth_url


# ─── TOKEN EXCHANGE ──────────────────────────────────────────
def exchange_code_for_token(code: str) -> dict:
    """
    Exchange authorization code for access token.
    This is the step Streamlit couldn't handle —
    it requires a stable HTTP endpoint to receive the callback.
    FastAPI handles this naturally with a route.
    """
    msal_app = get_msal_app()
    result = msal_app.acquire_token_by_authorization_code(
        code=code,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )

    if "error" in result:
        raise HTTPException(
            status_code=401,
            detail=f"OAuth error: {result.get('error_description')}"
        )

    return result


# ─── USER INFO FROM TOKEN ─────────────────────────────────────
def get_user_from_token(token_result: dict) -> dict:
    """Extract user info from the token result."""
    id_token_claims = token_result.get("id_token_claims", {})
    return {
        "ms_user_id":   id_token_claims.get("oid", ""),
        "email":        id_token_claims.get("email") or
                        id_token_claims.get("preferred_username", ""),
        "display_name": id_token_claims.get("name", "")
    }


# ─── DEPENDENCY — CURRENT USER ───────────────────────────────
async def get_current_user(
        token: str = Depends(security)) -> dict:
    """
    FastAPI dependency — validates token on every protected route.
    Usage: add `user = Depends(get_current_user)` to any endpoint.
    """
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated"
        )

    try:
        # For demo — decode without verification
        # Production: verify against Microsoft's JWKS endpoint
        payload = jwt.decode(
            token.credentials,
            options={"verify_signature": False}
        )
        return {
            "ms_user_id":   payload.get("oid", "demo"),
            "email":        payload.get("email", "demo@pdis.com"),
            "display_name": payload.get("name", "Demo User")
        }
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )


# ─── DEMO USER (no auth) ─────────────────────────────────────
async def get_demo_user() -> dict:
    """
    Use this dependency when auth is disabled for demo.
    Replace with get_current_user in production.
    """
    return {
        "ms_user_id":   "a0000000-0000-0000-0000-000000000001",
        "email":        "demo@pdis.com",
        "display_name": "Demo User"
    }