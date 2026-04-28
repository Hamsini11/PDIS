"""
security.py — Security middleware for PDIS
Handles rate limiting, input sanitization, audit logging.
"""

from fastapi import Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging
import time

# ─── RATE LIMITER ────────────────────────────────────────────
# Prevents token stuffing and abuse
limiter = Limiter(key_func=get_remote_address)

# ─── AUDIT LOGGER ────────────────────────────────────────────
# Every API call logged for 21 CFR Part 11 compliance
audit_logger = logging.getLogger("pdis.audit")
logging.basicConfig(
    filename="logs/audit.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)


def log_audit(
        user_id: str,
        action: str,
        resource: str,
        details: str = ""):
    """
    21 CFR Part 11 compliant audit trail.
    Every document access and query is logged.
    """
    audit_logger.info(
        f"user={user_id} | action={action} | "
        f"resource={resource} | details={details}"
    )


async def audit_middleware(request: Request, call_next):
    """Log all API requests for audit trail."""
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)

    audit_logger.info(
        f"method={request.method} | "
        f"path={request.url.path} | "
        f"status={response.status_code} | "
        f"duration={duration}ms"
    )
    return response


def validate_filename(filename: str) -> str:
    """
    Prevent path traversal attacks.
    e.g. user sends '../../etc/passwd' as filename
    """
    import os
    # remove any path components
    safe_name = os.path.basename(filename)
    if safe_name != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename"
        )
    # only allow PDF files
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )
    return safe_name


def check_file_size(content: bytes, max_mb: int = 50) -> None:
    """Prevent oversized file uploads."""
    max_bytes = max_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {max_mb}MB"
        )