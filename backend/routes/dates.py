"""
routes/dates.py
GET /api/dates → get extracted dates with optional filters
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))

from fastapi import APIRouter, Depends, Query
from typing import Optional
from backend.models import DateListResponse, DateEntry
from backend.auth import get_demo_user
from vector_store import get_all_dates

router = APIRouter()


@router.get("/dates", response_model=DateListResponse)
async def get_dates(
        filter_doc: Optional[str] = Query(None),
        flagged_only: bool = Query(False),
        user: dict = Depends(get_demo_user)):
    """
    Get extracted dates with optional filters.
    
    filter_doc: restrict to specific document
    flagged_only: return only compliance-flagged dates
    """
    dates = get_all_dates(
        filter_doc=filter_doc,
        flagged_only=flagged_only
    )

    date_entries = []
    for d in dates:
        date_entries.append(DateEntry(
            raw_date=  d.get("raw_date", ""),
            normalized=d.get("normalized"),
            page=      d.get("page", 0),
            doc=       d.get("doc", ""),
            flagged=   d.get("flagged", False),
            context=   d.get("text", "")[:100]
        ))

    flagged_count = sum(1 for d in date_entries if d.flagged)

    return DateListResponse(
        dates=date_entries,
        total=len(date_entries),
        flagged_count=flagged_count
    )