"""
routes/documents.py
POST /api/upload   → upload and index a PDF
GET  /api/documents → list all indexed documents
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from backend.models import DocumentListResponse
from backend.auth import get_demo_user
from backend.security import validate_filename, check_file_size, log_audit
from vector_store import list_documents

router = APIRouter()


@router.post("/upload")
async def upload_document(
        file: UploadFile = File(...),
        user: dict = Depends(get_demo_user)):
    """
    Upload and index a PDF document.
    Validates filename and file size before processing.
    """
    try:
        # Security checks
        safe_name = validate_filename(file.filename)
        content   = await file.read()
        check_file_size(content, max_mb=50)

        # Save to disk
        save_path = ROOT / "data" / "sample_docs" / safe_name
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.exists():
            return {
                "message":  f"Already indexed: {safe_name}",
                "filename": safe_name,
                "status":   "existing"
            }

        save_path.write_bytes(content)

        # Index the document
        from pipeline import run_compliance_scan
        run_compliance_scan(save_path)

        log_audit(
            user_id=user["ms_user_id"],
            action="UPLOAD",
            resource=safe_name
        )

        return {
            "message":  f"Successfully indexed: {safe_name}",
            "filename": safe_name,
            "status":   "indexed"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=DocumentListResponse)
async def get_documents(user: dict = Depends(get_demo_user)):
    """List all indexed documents."""
    docs = list_documents()
    return DocumentListResponse(
        documents=docs,
        total=len(docs)
    )