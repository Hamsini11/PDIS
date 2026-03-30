import os
import json
from typing import Optional
from pydantic import BaseModel, Field
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()

# ─── SCHEMA ──────────────────────────────────────────────────
class DateEntry(BaseModel):
    raw: str = Field(description="Exact date string as it appears in the document")
    normalized: Optional[str] = Field(default=None, description="ISO 8601 format: YYYY-MM-DD. If ambiguous, use best guess and flag it.")
    context: str = Field(description="The sentence or phrase surrounding the date (max 100 chars)")
    page: int = Field(description="Page number where the date was found")
    ambiguous: bool = Field(description="True if DD/MM vs MM/DD is unclear or date validity is uncertain")
    confidence: float = Field(description="0.0 to 1.0 — how confident you are this is a real, valid date")
    is_duration: bool = Field(default=False, description="True if this is a duration/period (e.g. '6 months') not a specific date")
    is_fiscal: bool = Field(default=False, description="True if this is a fiscal year reference (e.g. FY2023)")

class DocumentMetadata(BaseModel):
    document_title: Optional[str] = Field(default=None, description="Title of the document if present")
    document_type: Optional[str] = Field(default=None, description="e.g. DSCSA Report, FDA Guidance, SDF, Vendor Compliance")
    vendor: Optional[str] = Field(default=None, description="Vendor or organization name if present")
    version: Optional[str] = Field(default=None, description="Document version number if present")
    dates: list[DateEntry] = Field(default_factory=list, description="All dates found in the document")

# ─── EXTRACTION ───────────────────────────────────────────────
SYSTEM_PROMPT = """You are a pharmaceutical document intelligence specialist.
Your job is to extract structured information from regulatory and vendor compliance documents.

Rules:
- Only extract information that is EXPLICITLY present in the text
- For every date you extract, quote the surrounding context verbatim
- If a date is ambiguous (DD/MM vs MM/DD both valid), set ambiguous=true
- If a field is not present, return null — NEVER guess or infer
- Confidence score: 1.0 = unambiguous, 0.7 = likely correct, <0.5 = flag for review
- Normalize all dates to ISO 8601 (YYYY-MM-DD)
- Return ONLY valid JSON matching the schema. No preamble, no explanation."""

def extract_from_page(page_text: str, page_num: int, doc_title: str = "") -> list[DateEntry]:
    """Extract dates from a single page using Claude Sonnet."""

    if not page_text.strip():
        return []

    prompt = f"""Extract all dates from the following document page.
Document: {doc_title}
Page: {page_num}

Page text:
\"\"\"
{page_text[:3000]}
\"\"\"

Return a JSON array of date objects. Each object must have:
- raw: exact text as found
- normalized: ISO 8601 (YYYY-MM-DD)
- context: surrounding sentence (max 100 chars)
- page: {page_num}
- ambiguous: true/false
- confidence: 0.0-1.0

If no dates found, return empty array [].
Return ONLY the JSON array, nothing else."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    # strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
        return [DateEntry(**d) for d in parsed]
    except Exception as e:
        print(f"\n    ⚠️  Parse error on page {page_num}: {e}")
        print(f"    Raw response preview: {raw[:200]}")
        return []

def extract_document_metadata(full_text: str, filename: str) -> DocumentMetadata:
    """Extract top-level document metadata from first ~2000 chars."""

    prompt = f"""Extract metadata from this pharmaceutical document.
Filename: {filename}

Document text (first section):
\"\"\"
{full_text[:2000]}
\"\"\"

Return a JSON object with:
- document_title: string or null
- document_type: string or null  
- vendor: string or null
- version: string or null
- dates: [] (empty array — dates handled separately)

Return ONLY the JSON object, nothing else."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed = json.loads(raw)
    return DocumentMetadata(**parsed)


def run_extraction(page_texts: list[tuple[int, str]], filename: str, full_text: str) -> DocumentMetadata:
    """
    Full extraction pipeline for one document.
    page_texts: list of (page_num, text) tuples from main.py
    Returns DocumentMetadata with all dates populated.
    """
    print(f"\n  Extracting metadata...")
    metadata = extract_document_metadata(full_text, filename)
    metadata.dates = []

    print(f"  Extracting dates page by page...")
    for page_num, page_text in page_texts:
        if not page_text.strip():
            continue
        print(f"    Page {page_num}...", end=" ")
        dates = extract_from_page(page_text, page_num, metadata.document_title or filename)
        metadata.dates.extend(dates)
        print(f"{len(dates)} dates found")

    # validation pass — flag low confidence
    flagged = [d for d in metadata.dates if d.confidence < 0.7 or d.ambiguous]
    print(f"\n  Total dates: {len(metadata.dates)} | Flagged for review: {len(flagged)}")

    return metadata