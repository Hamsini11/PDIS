"""
main.py — FastAPI Application Entry Point
PDIS Backend API

Run with: uvicorn backend.main:app --reload --port 8000
Docs at:  http://localhost:8000/docs
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent

# Add project root to path so we can import core modules
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from slowapi.errors import RateLimitExceeded

from backend.security import limiter, audit_middleware
from backend.auth import get_auth_url, exchange_code_for_token, get_user_from_token
from supabase_client import get_or_create_user, new_session_id

# ─── APP SETUP ───────────────────────────────────────────────
app = FastAPI(
    title="PDIS — Pharmaceutical Document Intelligence System",
    description="AI-powered vendor document review for Pfizer compliance teams",
    version="0.4.0",
    docs_url="/docs",      # Swagger UI — great for demos
    redoc_url="/redoc"     # Alternative docs
)

# Rate limiter
app.state.limiter = limiter

# ─── MIDDLEWARE ──────────────────────────────────────────────
# CORS — allows Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501",
                   "http://localhost:8502",
                   "https://pdis-pfizer.streamlit.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Audit logging middleware
app.middleware("http")(audit_middleware)

# Rate limit error handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please slow down."}
    )


# ─── ROUTES — import here ────────────────────────────────────
from backend.routes import chat, documents, compare, dates
app.include_router(chat.router,      prefix="/api", tags=["Chat"])
app.include_router(documents.router, prefix="/api", tags=["Documents"])
app.include_router(compare.router,   prefix="/api", tags=["Comparison"])
app.include_router(dates.router,     prefix="/api", tags=["Dates"])


# ─── AUTH ROUTES ─────────────────────────────────────────────
@app.get("/auth/login")
async def login():
    """Redirect to Microsoft login page."""
    auth_url = get_auth_url()
    return RedirectResponse(url=auth_url)


@app.get("/auth/callback")
async def auth_callback(code: str, state: str = ""):
    """
    MS OAuth callback — this is why we needed FastAPI.
    Streamlit couldn't handle this stable endpoint.
    Microsoft redirects here after login with the auth code.
    We exchange it for a token and create/get the user.
    """
    # Exchange code for token
    token_result = exchange_code_for_token(code)

    # Get user info from token
    user_info = get_user_from_token(token_result)

    # Create or get user in Supabase
    user = get_or_create_user(
        ms_user_id=user_info["ms_user_id"],
        email=user_info["email"],
        display_name=user_info["display_name"]
    )

    # Return token to frontend
    # In production: set as HttpOnly cookie
    return {
        "access_token": token_result.get("access_token"),
        "user_id":      user["id"],
        "email":        user_info["email"],
        "display_name": user_info["display_name"],
        "session_id":   new_session_id()
    }


@app.get("/auth/logout")
async def logout():
    """Logout and clear session."""
    return {"message": "Logged out successfully"}


# ─── HEALTH CHECK ────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check endpoint for deployment monitoring."""
    return {
        "status": "healthy",
        "version": "0.4.0",
        "service": "PDIS Backend"
    }


@app.get("/")
async def root():
    return {
        "message": "PDIS API is running",
        "docs": "/docs",
        "health": "/health"
    }