import asyncio
from anthropic import Anthropic, AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

# ─── MODEL ROUTING ───────────────────────────────────────────
# Sonnet  → date extraction (accuracy critical)
# Haiku   → section detection + summarization (cost efficient, ~20x cheaper)
SONNET = "claude-sonnet-4-5"
HAIKU  = "claude-haiku-4-5-20251001"

# Two clients:
# sync  → single one-off calls
# async → parallel page processing (all pages fired simultaneously)
client       = Anthropic()
async_client = AsyncAnthropic()

# ─── TOOL SCHEMAS (native structured output) ─────────────────
# Claude is forced to call these tools — output is always valid.
# No JSON parsing, no try/except, no markdown fence stripping.

DATE_EXTRACTION_TOOL = {
    "name": "extract_dates",
    "description": "Extract all dates and temporal references from a document page",
    "input_schema": {
        "type": "object",
        "properties": {
            "dates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "raw":         {"type": "string",
                                        "description": "Exact text as found in document"},
                        "normalized":  {"type": ["string", "null"],
                                        "description": "ISO 8601 YYYY-MM-DD or null if not a calendar date"},
                        "context":     {"type": "string",
                                        "description": "Surrounding sentence, max 100 chars"},
                        "page":        {"type": "integer"},
                        "ambiguous":   {"type": "boolean",
                                        "description": "True if DD/MM vs MM/DD is unclear"},
                        "confidence":  {"type": "number",
                                        "description": "0.0 to 1.0"},
                        "is_duration": {"type": "boolean",
                                        "description": "True if this is a duration e.g. 6 months"},
                        "is_fiscal":   {"type": "boolean",
                                        "description": "True if this is a fiscal reference e.g. FY2023"}
                    },
                    "required": ["raw", "context", "page",
                                 "ambiguous", "confidence",
                                 "is_duration", "is_fiscal"]
                }
            }
        },
        "required": ["dates"]
    }
}

SECTION_DETECTION_TOOL = {
    "name": "detect_sections",
    "description": "Detect distinct document types within a vendor packet",
    "input_schema": {
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "section_type": {"type": "string",
                                         "description": "e.g. Certificate of Compliance, Vendor Contract, MSDS, Version Update"},
                        "start_page":   {"type": "integer"},
                        "end_page":     {"type": "integer"},
                        "confidence":   {"type": "number"}
                    },
                    "required": ["section_type", "start_page", "end_page", "confidence"]
                }
            }
        },
        "required": ["sections"]
    }
}

# ─── SYSTEM PROMPTS ──────────────────────────────────────────
EXTRACTION_SYSTEM = """You are a pharmaceutical document compliance specialist.

Rules:
- Extract ONLY dates explicitly present in the text
- Quote the surrounding context verbatim (max 100 chars)
- If a date is ambiguous (DD/MM vs MM/DD both possible), set ambiguous=true
- If not present or uncertain, return null for normalized — NEVER guess
- Confidence: 1.0 = unambiguous calendar date, 0.7 = likely correct, <0.5 = uncertain
- Normalize all calendar dates to ISO 8601 (YYYY-MM-DD)
- Durations like "6 months" → is_duration=true, normalized=null
- Fiscal refs like "FY2023" → is_fiscal=true, normalized=null"""

SECTION_SYSTEM = """You are a pharmaceutical document analyst.
Identify distinct document types within vendor submission packets.
Common types: Certificate of Compliance, Vendor Contract Agreement,
Material Safety Data Sheet, Version Update Notice, Test Report,
Regulatory Filing, Audit Report, Quality Agreement."""

SUMMARY_SYSTEM = """You are a pharmaceutical document reviewer.
Write concise summaries for busy Pfizer compliance reviewers.
Focus on: key dates, compliance status, version changes, action items.
Be direct. Flag anything requiring attention with a warning."""

# ─── ASYNC DATE EXTRACTION (parallel) ────────────────────────
async def extract_page_async(page_text: str, page_num: int,
                              doc_title: str = "",
                              semaphore=None) -> list[dict]:
    if not page_text.strip():
        return []

    prompt = f"""Extract all dates from this pharmaceutical document page.
Document: {doc_title}
Page: {page_num}

Page text:
\"\"\"
{page_text[:2500]}
\"\"\""""

    for attempt in range(3):
        try:
            async with semaphore:
                response = await async_client.messages.create(
                    model=HAIKU,
                    max_tokens=1024,
                    system=EXTRACTION_SYSTEM,
                    tools=[DATE_EXTRACTION_TOOL],
                    tool_choice={"type": "tool", "name": "extract_dates"},
                    messages=[{"role": "user", "content": prompt}]
                )
            result = response.content[0].input
            dates = result.get("dates", [])
            for d in dates:
                d["page"] = page_num
            return dates

        except Exception as e:
            if "rate_limit" in str(e) and attempt < 2:
                wait = (attempt + 1) * 20  # 20s, then 40s
                print(f"    Rate limit page {page_num}, waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"    Error page {page_num}: {e}")
                return []

    return []

async def extract_all_pages_parallel(
        page_texts: list[tuple[int, str]],
        doc_title: str = "",
        semaphore=None) -> list[dict]:
    """
    Fire ALL page extractions simultaneously.

    Sequential:  53 pages x 2s = ~106 seconds
    Parallel:    53 pages all at once = ~5-8 seconds
    """
    print(f"  Extracting dates from {len(page_texts)} pages in parallel...")

    tasks = [
        extract_page_async(text, num, doc_title, semaphore)
        for num, text in page_texts
        if text.strip()
    ]

    # All tasks fire at once, results collected when all complete
    results = await asyncio.gather(*tasks)

    # Flatten list of lists
    all_dates = [date for page_dates in results for date in page_dates]

    flagged = [d for d in all_dates
               if d.get("confidence", 1.0) < 0.7 or d.get("ambiguous")]

    print(f"  Total dates: {len(all_dates)} | Flagged: {len(flagged)}")
    return all_dates


# ─── SECTION DETECTION (Haiku) ───────────────────────────────
async def detect_packet_sections(full_text: str, filename: str,
                                   total_pages: int) -> list[dict]:
    """
    Detect document types within a vendor packet.
    Uses Haiku — ~20x cheaper than Sonnet, sufficient for classification.
    """
    print("  Detecting document sections (Haiku)...")

    # Sample beginning + end to catch headers and footers
    sample = full_text[:4000] + "\n...\n" + full_text[-1000:]

    prompt = f"""Analyze this vendor submission packet and identify distinct document sections.
Filename: {filename}
Total pages: {total_pages}

Document sample:
\"\"\"
{sample}
\"\"\""""

    response = await async_client.messages.create(
        model=HAIKU,
        max_tokens=1000,
        system=SECTION_SYSTEM,
        tools=[SECTION_DETECTION_TOOL],
        tool_choice={"type": "tool", "name": "detect_sections"},
        messages=[{"role": "user", "content": prompt}]
    )

    sections = response.content[0].input.get("sections", [])

    # Fallback if nothing detected
    if not sections:
        sections = [{
            "section_type": "Full Document",
            "start_page": 1,
            "end_page": total_pages,
            "confidence": 1.0
        }]

    print(f"  Found {len(sections)} section(s): "
          f"{[s['section_type'] for s in sections]}")
    return sections


# ─── SECTION SUMMARIZER (Haiku) ──────────────────────────────
async def summarize_section(section_text: str, section_type: str,
                             dates_in_section: list[dict]) -> str:
    """Summarize one document section using Haiku."""

    date_context = ""
    if dates_in_section:
        date_lines = [
            f"- {d['raw']} (page {d['page']}): {d['context'][:60]}"
            for d in dates_in_section[:10]
        ]
        date_context = "Key dates found:\n" + "\n".join(date_lines)

    prompt = f"""Summarize this {section_type} for a Pfizer compliance reviewer.

{date_context}

Document text:
\"\"\"
{section_text[:3000]}
\"\"\"

Write 2-3 sentences: purpose, key dates, compliance status, action items.
Flag anything requiring attention."""

    response = await async_client.messages.create(
        model=HAIKU,
        max_tokens=300,
        system=SUMMARY_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


# ─── PACKET SUMMARY ROLLUP (Haiku) ───────────────────────────
async def summarize_packet(filename: str, sections: list[dict],
                            section_summaries: dict,
                            all_dates: list[dict]) -> str:
    """Roll up all section summaries into one executive summary."""

    flagged_dates = [d for d in all_dates
                     if d.get("confidence", 1.0) < 0.7 or d.get("ambiguous")]

    sections_text = "\n".join([
        f"- {s['section_type']} (pages {s['start_page']}-{s['end_page']}): "
        f"{section_summaries.get(s['section_type'], 'No summary')}"
        for s in sections
    ])

    prompt = f"""Create a packet-level summary for this vendor submission.

Filename: {filename}
Total dates extracted: {len(all_dates)}
Flagged for review: {len(flagged_dates)}

Sections:
{sections_text}

Write a 3-4 sentence executive summary. Lead with any flags or action items."""

    response = await async_client.messages.create(
        model=HAIKU,
        max_tokens=400,
        system=SUMMARY_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


# ─── MAIN ORCHESTRATOR ───────────────────────────────────────
def run_extraction(page_texts: list[tuple[int, str]],
                   filename: str,
                   full_text: str) -> dict:
    """
    Full extraction pipeline for one document.
    Synchronous wrapper around the async implementation.
    Called from main.py

    Returns dict with:
      dates             → all extracted dates
      sections          → detected document sections
      section_summaries → Haiku summary per section
      packet_summary    → executive summary for reviewer
      flagged_count     → number of dates needing review
    """
    return asyncio.run(_run_async(page_texts, filename, full_text))


async def _run_async(page_texts: list[tuple[int, str]],
                     filename: str, full_text: str) -> dict:
    """Async implementation — orchestrates all steps."""
    
    semaphore = asyncio.Semaphore(3)
    total_pages = len(page_texts)
    doc_title   = filename.replace(".pdf", "").replace("_", " ")

    # Step 1 — Parallel date extraction (Sonnet, all pages at once)
    all_dates = await extract_all_pages_parallel(page_texts, doc_title, semaphore)

    # Step 2 — Section detection (Haiku)
    sections = await detect_packet_sections(full_text, filename, total_pages)

    # Step 3 — Summarize each section in parallel (Haiku)
    print("  Summarizing sections (Haiku)...")
    page_lookup = {num: text for num, text in page_texts}

    async def summarize_one(section: dict) -> tuple[str, str]:
        section_text = " ".join([
            page_lookup.get(p, "")
            for p in range(section["start_page"], section["end_page"] + 1)
        ])
        dates_in_section = [
            d for d in all_dates
            if section["start_page"] <= d.get("page", 0) <= section["end_page"]
        ]
        summary = await summarize_section(
            section_text, section["section_type"], dates_in_section
        )
        return section["section_type"], summary

    summary_results   = await asyncio.gather(*[summarize_one(s) for s in sections])
    section_summaries = dict(summary_results)

    # Step 4 — Packet-level rollup (Haiku)
    print("  Generating packet summary (Haiku)...")
    packet_summary = await summarize_packet(
        filename, sections, section_summaries, all_dates
    )

    print(f"\n  Packet summary preview:\n  {packet_summary[:200]}...\n")

    return {
        "dates":             all_dates,
        "sections":          sections,
        "section_summaries": section_summaries,
        "packet_summary":    packet_summary,
        "flagged_count":     sum(1 for d in all_dates
                                 if d.get("confidence", 1.0) < 0.7
                                 or d.get("ambiguous"))
    }