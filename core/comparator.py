import asyncio
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from core.vector_store import get_all_dates, list_documents, search

load_dotenv()

async_client = AsyncAnthropic()
SONNET = "claude-sonnet-4-5"
HAIKU  = "claude-haiku-4-5-20251001"

# ─── TOOL SCHEMA ─────────────────────────────────────────────
COMPARISON_TOOL = {
    "name": "compare_documents",
    "description": "Compare two pharmaceutical document versions and output a structured diff",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "2-3 sentence executive summary of what changed"
            },
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field":     {"type": "string",
                                      "description": "Field name e.g. Effective Date"},
                        "doc_a":     {"type": "string",
                                      "description": "Value in Doc A"},
                        "doc_b":     {"type": "string",
                                      "description": "Value in Doc B"},
                        "source_a":  {"type": "string",
                                      "description": "Page number and context from Doc A"},
                        "source_b":  {"type": "string", 
                                      "description": "Page number and context from Doc B"},
                        "changed":   {"type": "boolean"},
                        "flag":      {"type": "boolean",
                                      "description": "True if this change is a compliance concern"}
                    },
                    "required": ["field", "doc_a", "doc_b", "source_a", "source_b", "changed", "flag"]
                }
            }
        },
        "required": ["summary", "fields"]
    }
}

# ─── STEP 1: GET STRUCTURED DATA FROM INDEX ──────────────────
def get_doc_data_from_index(filename: str) -> dict:
    """
    Pull all structured data for a doc from FAISS index.
    Zero API calls — uses what we already extracted.
    """
    dates = get_all_dates(filter_doc=filename)
    flagged = [d for d in dates if d.get("flagged")]

    # get section info via semantic search
    section_chunks = search(
        "document section type certificate compliance contract",
        top_k=5,
        filter_doc=filename
    )

    # extract unique section types mentioned
    sections = list({
        c.get("text", "")[:50]
        for c in section_chunks
        if c.get("is_date") is not True
    })

    # get calendar dates only (not durations/fiscal)
    calendar_dates = [
        d for d in dates
        if d.get("normalized") and
        not d.get("is_duration") and
        not d.get("is_fiscal")
    ]

    # find earliest and latest
    normalized = sorted([
        d["normalized"] for d in calendar_dates
        if d.get("normalized")
    ])

    return {
        "filename":      filename,
        "total_dates":   len(dates),
        "calendar_dates": len(calendar_dates),
        "earliest_date": normalized[0] if normalized else "N/A",
        "latest_date":   normalized[-1] if normalized else "N/A",
        "flagged_count": len(flagged),
        "flagged_dates": [d["raw_date"] for d in flagged[:5]],
        "sections":      sections[:5],
        "date_sample":   [d["raw_date"] for d in calendar_dates[:8]]
    }


# ─── STEP 2: SUMMARIZE FRESH DOC (Haiku) ─────────────────────
async def summarize_for_comparison(
        page_texts: list[tuple[int, str]],
        filename: str) -> str:
    """
    Lightweight Haiku summary of a fresh doc for comparison.
    Only called when doc is NOT already indexed.
    """
    # take first 3000 chars of full text
    full_text = " ".join([t for _, t in page_texts])[:3000]

    response = await async_client.messages.create(
        model=HAIKU,
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""Summarize this pharmaceutical document for comparison.
Extract: document type, key dates, sections present, any compliance flags.
Be concise — this is for diff comparison only.

Document: {filename}
Text:
\"\"\"
{full_text}
\"\"\""""
        }]
    )
    return response.content[0].text


# ─── STEP 3: COMPARE (Sonnet) ────────────────────────────────
async def compare_documents(
        data_a: dict, data_b: dict,
        summary_a: str = "", summary_b: str = "") -> dict:
    """
    Sonnet compares structured data from both docs.
    Returns structured diff via native tool output.
    """

    prompt = f"""Compare these two pharmaceutical document versions.
Identify what changed, what's new, and any compliance concerns.

DOC A: {data_a['filename']}
- Total dates: {data_a['total_dates']}
- Calendar dates: {data_a['calendar_dates']}
- Earliest date: {data_a['earliest_date']}
- Latest date: {data_a['latest_date']}
- Flagged: {data_a['flagged_count']} ({', '.join(data_a['flagged_dates']) or 'none'})
- Key dates: {', '.join(data_a['date_sample']) or 'none'}
- Sections: {', '.join(data_a['sections']) or 'not detected'}
{f'- Summary: {summary_a}' if summary_a else ''}

DOC B: {data_b['filename']}
- Total dates: {data_b['total_dates']}
- Calendar dates: {data_b['calendar_dates']}
- Earliest date: {data_b['earliest_date']}
- Latest date: {data_b['latest_date']}
- Flagged: {data_b['flagged_count']} ({', '.join(data_b['flagged_dates']) or 'none'})
- Key dates: {', '.join(data_b['date_sample']) or 'none'}
- Sections: {', '.join(data_b['sections']) or 'not detected'}
{f'- Summary: {summary_b}' if summary_b else ''}

Generate a comprehensive diff table covering:
- Document type
- Effective/key dates
- Date count changes
- Section changes
- Compliance flag changes
- Content summary changes
Flag any changes that could indicate compliance risk."""

    response = await async_client.messages.create(
        model=HAIKU,
        max_tokens=2000,
        tools=[COMPARISON_TOOL],
        tool_choice={"type": "tool", "name": "compare_documents"},
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].input


# ─── MAIN ORCHESTRATOR ────────────────────────────────────────
def run_comparison(
        filename_a: str,
        filename_b: str,
        page_texts_a: list[tuple[int, str]] = None,
        page_texts_b: list[tuple[int, str]] = None) -> dict:
    """
    Main entry point for version comparison.
    Handles all 3 cases:

    Case 1: Both indexed → 1 Sonnet call (fastest)
    Case 2: A indexed, B fresh → 1 Haiku + 1 Sonnet
    Case 3: Both fresh → 1 Haiku + 1 Haiku + 1 Sonnet (map-reduce)

    filename_a:    doc A name (v1 — older version)
    filename_b:    doc B name (v2 — newer version)
    page_texts_a:  only needed if A is NOT indexed
    page_texts_b:  only needed if B is NOT indexed

    Returns:
      summary:    executive summary string
      fields:     list of diff rows for table display
      api_calls:  how many API calls were made
    """
    return asyncio.run(_compare_async(
        filename_a, filename_b, page_texts_a, page_texts_b
    ))


async def _compare_async(
        filename_a: str,
        filename_b: str,
        page_texts_a: list[tuple[int, str]] = None,
        page_texts_b: list[tuple[int, str]] = None) -> dict:

    indexed_docs = list_documents()
    api_calls = 0

    print(f"\n  Comparing: {filename_a} vs {filename_b}")

    # ── Step 1: Get Doc A data ────────────────────────────────
    if filename_a in indexed_docs:
        # Case 1 or 2 — pull from index, zero API calls
        print(f"  Doc A: pulling from index (0 API calls)")
        data_a   = get_doc_data_from_index(filename_a)
        summary_a = ""
    elif page_texts_a:
        # Case 3 — fresh upload, summarize with Haiku
        print(f"  Doc A: fresh upload — summarizing (1 Haiku call)")
        summary_a = await summarize_for_comparison(page_texts_a, filename_a)
        data_a = {
            "filename":      filename_a,
            "total_dates":   0,
            "calendar_dates": 0,
            "earliest_date": "N/A",
            "latest_date":   "N/A",
            "flagged_count": 0,
            "flagged_dates": [],
            "sections":      [],
            "date_sample":   []
        }
        api_calls += 1
    else:
        raise ValueError(
            f"{filename_a} is not indexed and no page texts provided. "
            f"Please index it first or upload it for comparison."
        )

    # ── Step 2: Get Doc B data ────────────────────────────────
    if filename_b in indexed_docs:
        # pull from index, zero API calls
        print(f"  Doc B: pulling from index (0 API calls)")
        data_b   = get_doc_data_from_index(filename_b)
        summary_b = ""
    elif page_texts_b:
        # fresh upload — one Haiku call
        print(f"  Doc B: fresh upload — summarizing (1 Haiku call)")
        summary_b = await summarize_for_comparison(page_texts_b, filename_b)
        data_b = {
            "filename":      filename_b,
            "total_dates":   0,
            "calendar_dates": 0,
            "earliest_date": "N/A",
            "latest_date":   "N/A",
            "flagged_count": 0,
            "flagged_dates": [],
            "sections":      [],
            "date_sample":   []
        }
        api_calls += 1
    else:
        raise ValueError(
            f"{filename_b} is not indexed and no page texts provided. "
            f"Please index it first or upload it for comparison."
        )

    # Step 3 — Compare with Sonnet (1 API call)
    print(f"  Comparing with Sonnet (1 API call)...")
    result = await compare_documents(data_a, data_b, summary_a, summary_b)
    api_calls += 1

    print(f"  Done! Total API calls: {api_calls}")
    print(f"  Changes found: {sum(1 for f in result['fields'] if f['changed'])}")
    print(f"  Compliance flags: {sum(1 for f in result['fields'] if f['flag'])}")

    return {
        "doc_a":       filename_a,
        "doc_b":       filename_b,
        "summary":     result["summary"],
        "fields":      result["fields"],
        "api_calls":   api_calls
    }