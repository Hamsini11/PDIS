"""
routes/compare.py
POST /api/compare → compare two document versions
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from typing import Optional
from backend.models import CompareResponse
from backend.auth import get_demo_user
from backend.security import log_audit
import tempfile

router = APIRouter()


@router.post("/compare", response_model=CompareResponse)
async def compare_documents(
        filename_a: str = Form(...),
        filename_b: str = Form(...),
        file_a: Optional[UploadFile] = File(None),
        file_b: Optional[UploadFile] = File(None),
        user: dict = Depends(get_demo_user)):
    """
    Compare two document versions.
    Accepts indexed filenames OR fresh uploads for either doc.
    """
    try:
        from comparator import run_comparison

        # Resolve Doc A
        texts_a = None
        if file_a:
            content_a = await file_a.read()
            with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf") as f:
                f.write(content_a)
                tmp_a = f.name
            from pipeline import extract_text_from_pdf
            _, texts_a = extract_text_from_pdf(tmp_a)

        # Resolve Doc B
        texts_b = None
        if file_b:
            content_b = await file_b.read()
            with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf") as f:
                f.write(content_b)
                tmp_b = f.name
            from pipeline import extract_text_from_pdf
            _, texts_b = extract_text_from_pdf(tmp_b)

        if filename_a == filename_b:
            raise ValueError("Please select two different documents")

        result = run_comparison(
            filename_a, filename_b, texts_a, texts_b
        )

        log_audit(
            user_id=user["ms_user_id"],
            action="COMPARE",
            resource=f"{filename_a} vs {filename_b}"
        )

        return CompareResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))